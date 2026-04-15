/*
GPT-2 CUDA Inference - forward + sampling only, no backward/optimizer.
Supports KV cache for O(T) decode steps after an O(T^2) prefill.
*/

// TESTING skips main() in train_gpt2.cu so we can define our own
#define TESTING
#include "train_gpt2.cu"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int parse_tokens(const char* s, int* buf, int max_len) {
    int n = 0;
    char* tmp = strdup(s);
    char* tok = strtok(tmp, " ");
    while (tok && n < max_len) {
        buf[n++] = atoi(tok);
        tok = strtok(NULL, " ");
    }
    free(tmp);
    return n;
}

// ---------------------------------------------------------------------------
// KV Cache kernels
// ---------------------------------------------------------------------------

// Element-wise add: out[i] = inp1[i] + inp2[i].  out may alias inp1.
__global__ void simple_add_kernel(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (floatX)((float)inp1[i] + (float)inp2[i]);
}

// GeLU (tanh approximation) — no alignment requirement, safe for arbitrary N.
#define DECODE_GELU_SCALE sqrtf(2.0f / M_PI)
__global__ void decode_gelu_kernel(floatX* out, const floatX* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x = (float)inp[i];
    float cube = 0.044715f * x * x * x;
    out[i] = (floatX)(0.5f * x * (1.0f + tanhf(DECODE_GELU_SCALE * (x + cube))));
}

// Embed a single token at an arbitrary sequence position.
// out[C] = wte[token_id * C] + wpe[pos * C]
__global__ void encode_at_pos_kernel(floatX* out, int token_id, int pos,
                                     const floatX* wte, const floatX* wpe, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) out[c] = (floatX)((float)wte[token_id * C + c] + (float)wpe[pos * C + c]);
}

// Copy K or V from prefill qkvr layout [NH, T_fwd, HS] (source stride = T_fwd)
// into cache layout [NH, T_max, HS], copying only the first T_real token positions.
__global__ void kv_populate_kernel(floatX* dst, const floatX* src,
                                   int NH, int T_real, int T_max, int HS, int T_fwd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = NH * T_real * HS;
    if (idx >= total) return;
    int h = idx / (T_real * HS);
    int t = (idx / HS) % T_real;
    int d = idx % HS;
    dst[(size_t)h * T_max * HS + t * HS + d] = src[(size_t)h * T_fwd * HS + t * HS + d];
}

// Append a single new K or V vector [NH, HS] into cache [NH, T_max, HS] at position pos.
__global__ void kv_insert_kernel(floatX* cache_kv, const floatX* new_kv,
                                 int pos, int T_max, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NH * HS) return;
    int h = idx / HS;
    int d = idx % HS;
    cache_kv[(size_t)h * T_max * HS + pos * HS + d] = new_kv[h * HS + d];
}

// Single-query decode attention.
// One block per head.  smem = seq_len * sizeof(float).
//   q        : [NH, HS]
//   k_cache  : [NH, T_max, HS]  (positions 0..seq_len-1 are valid)
//   v_cache  : [NH, T_max, HS]
//   out      : [NH, HS]
__global__ void kv_decode_attn_kernel(floatX* out,
                                      const floatX* q,
                                      const floatX* k_cache,
                                      const floatX* v_cache,
                                      int seq_len, int T_max, int HS, float scale) {
    int h = blockIdx.x;
    const floatX* q_h = q         + h * HS;
    const floatX* k_h = k_cache   + (size_t)h * T_max * HS;
    const floatX* v_h = v_cache   + (size_t)h * T_max * HS;
    floatX*       o_h = out       + h * HS;

    extern __shared__ float smem[]; // float scores[seq_len]

    // Phase 1: compute scores Q . K[j] for each cached position j
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        float s = 0.0f;
        for (int d = 0; d < HS; d++) s += (float)q_h[d] * (float)k_h[j * HS + d];
        smem[j] = s * scale;
    }
    __syncthreads();

    // Phase 2: softmax (thread 0 only — seq_len <= T_max <= 1024)
    if (threadIdx.x == 0) {
        float maxv = smem[0];
        for (int j = 1; j < seq_len; j++) if (smem[j] > maxv) maxv = smem[j];
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) { smem[j] = expf(smem[j] - maxv); sum += smem[j]; }
        float inv = 1.0f / sum;
        for (int j = 0; j < seq_len; j++) smem[j] *= inv;
    }
    __syncthreads();

    // Phase 3: weighted sum over V
    for (int d = threadIdx.x; d < HS; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) acc += smem[j] * (float)v_h[j * HS + d];
        o_h[d] = (floatX)acc;
    }
}

// ---------------------------------------------------------------------------
// KVCache struct
// ---------------------------------------------------------------------------

typedef struct {
    floatX* k;   // [L, NH, T_max, HS]  GPU
    floatX* v;   // [L, NH, T_max, HS]  GPU
    int pos;     // tokens currently cached (= sequence length so far)
    int max_T;
} KVCache;

void kv_cache_init(KVCache* cache, GPT2* model) {
    int L     = model->config.num_layers;
    int NH    = model->config.num_heads;
    int C     = model->config.channels;
    int HS    = C / NH;
    int T_max = model->seq_len;
    size_t sz = (size_t)L * NH * T_max * HS * sizeof(floatX);
    cudaCheck(cudaMalloc(&cache->k, sz));
    cudaCheck(cudaMalloc(&cache->v, sz));
    cache->pos   = 0;
    cache->max_T = T_max;
}

void kv_cache_free(KVCache* cache) {
    cudaCheck(cudaFree(cache->k));
    cudaCheck(cudaFree(cache->v));
}

// ---------------------------------------------------------------------------
// DecodeBuffers: small GPU scratch pads for single-token decode
// ---------------------------------------------------------------------------

typedef struct {
    floatX* hidden;    // [C]
    floatX* ln_out;    // [C]   layernorm output, reused across layers
    floatX* qkv;       // [3*C] QKV matmul output (before permute)
    floatX* qkvr;      // [3*C] after permute: q=[0..C), k=[C..2C), v=[2C..3C)
    floatX* atty;      // [C]   decode attention output
    floatX* tmp;       // [C]   scratch: attn_proj / mlp_proj output
    floatX* fch;       // [4*C] MLP pre-gelu
    floatX* fch_gelu;  // [4*C] MLP post-gelu
    floatX* logits;    // [Vp]  final logits (floatX on GPU)
} DecodeBuffers;

void decode_buffers_init(DecodeBuffers* d, GPT2* model) {
    int C  = model->config.channels;
    int Vp = model->config.padded_vocab_size;
    // Allocate in one block for convenience
    size_t total = ((size_t)(1+1+3+3+1+1+4+4) * C + Vp) * sizeof(floatX);
    floatX* buf;
    cudaCheck(cudaMalloc(&buf, total));
    floatX* p = buf;
    d->hidden   = p; p += C;
    d->ln_out   = p; p += C;
    d->qkv      = p; p += 3*C;
    d->qkvr     = p; p += 3*C;
    d->atty     = p; p += C;
    d->tmp      = p; p += C;
    d->fch      = p; p += 4*C;
    d->fch_gelu = p; p += 4*C;
    d->logits   = p;
}

void decode_buffers_free(DecodeBuffers* d) {
    cudaCheck(cudaFree(d->hidden)); // frees the whole block
}

// ---------------------------------------------------------------------------
// kv_cache_prefill: populate cache from acts.qkvr after gpt2_forward(T_fwd).
// T_fwd  = the T passed to gpt2_forward (must be % 4 == 0, may be padded)
// T_real = number of actual prompt tokens to cache (T_real <= T_fwd)
// Call AFTER gpt2_forward(&model, tokens, B, T_fwd).
// ---------------------------------------------------------------------------

void kv_cache_prefill(KVCache* cache, GPT2* model, int T_fwd, int T_real) {
    int L     = model->config.num_layers;
    int B     = 1;
    int NH    = model->config.num_heads;
    int C     = model->config.channels;
    int HS    = C / NH;
    int T_max = cache->max_T;
    const int bs = 256;

    for (int l = 0; l < L; l++) {
        // After attention_forward's permute_kernel, qkvr for layer l is laid out
        // with stride T_fwd (the T passed to gpt2_forward):
        //   [B*T_fwd*C * 0]: Q  [B, NH, T_fwd, HS]
        //   [B*T_fwd*C * 1]: K  [B, NH, T_fwd, HS]  <- we copy first T_real positions
        //   [B*T_fwd*C * 2]: V  [B, NH, T_fwd, HS]
        floatX* src_k = model->acts.qkvr + (size_t)l * B * T_fwd * 3 * C
                                         + (size_t)B * T_fwd * C;
        floatX* src_v = model->acts.qkvr + (size_t)l * B * T_fwd * 3 * C
                                         + (size_t)2 * B * T_fwd * C;
        floatX* dst_k = cache->k + (size_t)l * NH * T_max * HS;
        floatX* dst_v = cache->v + (size_t)l * NH * T_max * HS;

        // Copy only T_real token positions (source stride = T_fwd, dest stride = T_max)
        int total = NH * T_real * HS;
        kv_populate_kernel<<<CEIL_DIV(total, bs), bs, 0, main_stream>>>(
            dst_k, src_k, NH, T_real, T_max, HS, T_fwd);
        kv_populate_kernel<<<CEIL_DIV(total, bs), bs, 0, main_stream>>>(
            dst_v, src_v, NH, T_real, T_max, HS, T_fwd);
    }
    cache->pos = T_real;
}

// ---------------------------------------------------------------------------
// kv_cache_decode_step: single-token forward with KV cache.
//
//   new_token : CPU-side integer — the token at position cache->pos
//   On return : d->logits holds GPU floatX logits for position cache->pos+1.
//               cache->pos is incremented.
// ---------------------------------------------------------------------------

void kv_cache_decode_step(GPT2* model, KVCache* cache, DecodeBuffers* d, int new_token) {
    const int L     = model->config.num_layers;
    const int NH    = model->config.num_heads;
    const int C     = model->config.channels;
    const int HS    = C / NH;
    const int Vp    = model->config.padded_vocab_size;
    const int pos   = cache->pos;       // position of new_token in the sequence
    const int T_max = cache->max_T;
    const float scale = 1.0f / sqrtf((float)HS);
    const int bs = 256;

    // 1. Encode: hidden = wte[new_token] + wpe[pos]
    encode_at_pos_kernel<<<CEIL_DIV(C, bs), bs, 0, main_stream>>>(
        d->hidden, new_token, pos, model->params.wte, model->params.wpe, C);

    // 2. LN1 for layer 0 (ln_out reused as the running "normalized" input to each layer)
    layernorm_forward(d->ln_out, nullptr, nullptr,
                      d->hidden, model->params.ln1w, model->params.ln1b,
                      1, 1, C, main_stream);

    for (int l = 0; l < L; l++) {
        floatX* l_qkvw     = model->params.qkvw     + (size_t)l * 3*C * C;
        floatX* l_qkvb     = model->params.qkvb     + (size_t)l * 3*C;
        floatX* l_attprojw = model->params.attprojw + (size_t)l * C * C;
        floatX* l_attprojb = model->params.attprojb + (size_t)l * C;
        floatX* l_ln2w     = model->params.ln2w     + (size_t)l * C;
        floatX* l_ln2b     = model->params.ln2b     + (size_t)l * C;
        floatX* l_fcw      = model->params.fcw      + (size_t)l * 4*C * C;
        floatX* l_fcb      = model->params.fcb      + (size_t)l * 4*C;
        floatX* l_fcprojw  = model->params.fcprojw  + (size_t)l * C * 4*C;
        floatX* l_fcprojb  = model->params.fcprojb  + (size_t)l * C;
        floatX* l_k_cache  = cache->k + (size_t)l * NH * T_max * HS;
        floatX* l_v_cache  = cache->v + (size_t)l * NH * T_max * HS;

        // QKV projection on the single new token
        matmul_forward_cublaslt(d->qkv, d->ln_out, l_qkvw, l_qkvb,
                                1, 1, C, 3*C, main_stream);

        // Permute [1,1,3,NH,HS] → q[NH,HS] k[NH,HS] v[NH,HS]
        {
            int total = NH * HS; // = C for B=T=1
            permute_kernel<<<CEIL_DIV(total, bs), bs, 0, main_stream>>>(
                d->qkvr,        // q
                d->qkvr + C,    // k
                d->qkvr + 2*C,  // v
                d->qkv, 1, 1, NH, HS);
        }

        // Append K and V to cache at position pos
        kv_insert_kernel<<<CEIL_DIV(C, bs), bs, 0, main_stream>>>(
            l_k_cache, d->qkvr + C,    pos, T_max, NH, HS);
        kv_insert_kernel<<<CEIL_DIV(C, bs), bs, 0, main_stream>>>(
            l_v_cache, d->qkvr + 2*C,  pos, T_max, NH, HS);

        // Decode attention: Q[NH,HS] × K_cache[0..pos] → out[NH,HS]
        int seq_len = pos + 1;
        size_t smem = (size_t)seq_len * sizeof(float);
        kv_decode_attn_kernel<<<NH, bs, smem, main_stream>>>(
            d->atty, d->qkvr, l_k_cache, l_v_cache,
            seq_len, T_max, HS, scale);

        // Attention projection
        matmul_forward_cublaslt(d->tmp, d->atty, l_attprojw, l_attprojb,
                                1, 1, C, C, main_stream);

        // Residual add + LN2
        simple_add_kernel<<<CEIL_DIV(C, bs), bs, 0, main_stream>>>(
            d->hidden, d->hidden, d->tmp, C);
        layernorm_forward(d->ln_out, nullptr, nullptr,
                          d->hidden, l_ln2w, l_ln2b, 1, 1, C, main_stream);

        // MLP: fc matmul then gelu separately
        // (gelu_forward in gelu.cuh requires N%4096==0 which fails for N=4*C=3072,
        //  so we use our own decode_gelu_kernel that handles arbitrary N)
        matmul_forward_cublaslt(d->fch, d->ln_out, l_fcw, l_fcb,
                                1, 1, C, 4*C, main_stream);
        decode_gelu_kernel<<<CEIL_DIV(4*C, bs), bs, 0, main_stream>>>(
            d->fch_gelu, d->fch, 4*C);

        // MLP: proj back to C
        matmul_forward_cublaslt(d->tmp, d->fch_gelu, l_fcprojw, l_fcprojb,
                                1, 1, 4*C, C, main_stream);

        // Residual add + LN for next layer (LN1[l+1]) or final LN
        simple_add_kernel<<<CEIL_DIV(C, bs), bs, 0, main_stream>>>(
            d->hidden, d->hidden, d->tmp, C);

        const floatX* next_lnw = (l + 1 < L)
            ? model->params.ln1w + (size_t)(l + 1) * C
            : model->params.lnfw;
        const floatX* next_lnb = (l + 1 < L)
            ? model->params.ln1b + (size_t)(l + 1) * C
            : model->params.lnfb;
        layernorm_forward(d->ln_out, nullptr, nullptr,
                          d->hidden, next_lnw, next_lnb, 1, 1, C, main_stream);
    }

    // 3. Unembedding: logits = ln_out @ wte^T  [Vp]
    matmul_forward_cublaslt(d->logits, d->ln_out, model->params.wte, nullptr,
                            1, 1, C, Vp, main_stream);

    cudaCheck(cudaDeviceSynchronize());
    cache->pos++;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
    const char* load_filename  = "gpt2_124M_bf16.bin";
    const char* prompt_tokens  = NULL;
    int genT                   = 256;
    int stop_at_eot            = 0;
    unsigned long long seed    = 1337;

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) { fprintf(stderr, "missing arg\n"); return 1; }
        if      (argv[i][1] == 'e') { load_filename = argv[i+1]; }
        else if (argv[i][1] == 'g') { genT = atoi(argv[i+1]); }
        else if (argv[i][1] == 'p') { prompt_tokens = argv[i+1]; }
        else if (argv[i][1] == 'x') { stop_at_eot = atoi(argv[i+1]); }
        else if (argv[i][1] == 's') { seed = (unsigned long long)atoll(argv[i+1]); }
    }

    multi_gpu_config = multi_gpu_config_init(1, 0, 1, (char*)"", (char*)"", (char*)"fs");
    common_start(false, false);

    GPT2 model;
    gpt2_init_common(&model);
    gpt2_build_from_checkpoint(&model, load_filename);

    int B = 1, T = 1024;
    model.batch_size = B;
    model.seq_len    = T;
    fill_in_activation_sizes(&model.acts, model.acts_specs, B, T, model.config, model.recompute);
    model.acts_memory = malloc_and_point_activations(model.acts_specs);
    cudaCheck(cudaMalloc((void**)&model.inputs,  B * T * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&model.targets, B * T * sizeof(int)));
    cudaCheck(cudaMallocHost((void**)&model.cpu_losses, B * T * sizeof(float)));

    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    int*    gen_tokens     = (int*)   mallocCheck(B * T * sizeof(int));
    floatX* cpu_logits_raw = (floatX*)mallocCheck(model.config.vocab_size * sizeof(floatX));
    float*  cpu_logits     = (float*) mallocCheck(model.config.vocab_size * sizeof(float));

    int eot_token  = tokenizer.eot_token;
    int prompt_len = 0;

    // fill buffer with eot, then overwrite with prompt if provided
    for (int i = 0; i < B * T; i++) gen_tokens[i] = eot_token;

    // init KV cache and decode buffers
    KVCache       cache;
    DecodeBuffers d;
    kv_cache_init(&cache, &model);
    decode_buffers_init(&d, &model);

    // ---- Prefill ----
    // Run gpt2_forward over the initial sequence (prompt, or just eot token),
    // then copy K/V into the cache.
    if (prompt_tokens != NULL) {
        prompt_len = parse_tokens(prompt_tokens, gen_tokens, T);
        if (prompt_len <= 0) { fprintf(stderr, "failed to parse prompt tokens\n"); return 1; }
        printf("prompt: ");
        for (int i = 0; i < prompt_len; i++) {
            if (tokenizer.init_ok) safe_printf(tokenizer_decode(&tokenizer, gen_tokens[i]));
            else                   printf("%d ", gen_tokens[i]);
        }
        printf("\n---\n");
    }

    int prefill_len = (prompt_len > 0) ? prompt_len : 1; // at least the initial eot token
    // softmax_forward_kernel5 requires T % 4 == 0; round up to next multiple of 256 like the
    // original generation loop did (this is also how gpt2_forward is typically called)
    int prefill_T = CEIL_DIV(prefill_len, min(T, 256)) * min(T, 256);
    gpt2_forward(&model, gen_tokens, B, prefill_T);
    kv_cache_prefill(&cache, &model, prefill_T, prefill_len);

    // Extract logits at the last REAL token position (prefill_len-1, not the padding)
    {
        floatX* logits_gpu = model.acts.output +
                             (size_t)(prefill_len - 1) * model.config.padded_vocab_size;
        cudaCheck(cudaMemcpy(cpu_logits_raw, logits_gpu,
                             model.config.vocab_size * sizeof(floatX),
                             cudaMemcpyDeviceToHost));
    }

    unsigned long long rng_state = seed;

    // ---- Generation loop ----
    printf("generating:\n---\n");
    for (int t = prefill_len; t < genT; t++) {
        if (cache.pos >= cache.max_T) {
            printf("\n[reached max sequence length %d]\n", cache.max_T);
            break;
        }

        // Convert logits to float and sample
        for (int i = 0; i < model.config.vocab_size; i++)
            cpu_logits[i] = (float)cpu_logits_raw[i];

        float coin      = random_f32(&rng_state);
        int next_token  = sample_softmax(cpu_logits, model.config.vocab_size, coin);
        gen_tokens[t]   = next_token;

        if (next_token == eot_token) {
            if (stop_at_eot) { break; }
            printf("<|endoftext|>");
            fflush(stdout);
        } else {
            if (tokenizer.init_ok) safe_printf(tokenizer_decode(&tokenizer, next_token));
            else                   printf("%d ", next_token);
            fflush(stdout);
        }

        // KV cache decode step: process next_token, produce logits for t+1
        kv_cache_decode_step(&model, &cache, &d, next_token);

        // Copy GPU logits to CPU for next iteration's sampling
        cudaCheck(cudaMemcpy(cpu_logits_raw, d.logits,
                             model.config.vocab_size * sizeof(floatX),
                             cudaMemcpyDeviceToHost));
    }
    printf("\n---\n");

    kv_cache_free(&cache);
    decode_buffers_free(&d);
    free(gen_tokens);
    free(cpu_logits_raw);
    free(cpu_logits);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    common_free(model);
    return 0;
}
