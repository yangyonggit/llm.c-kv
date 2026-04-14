/*
GPT-2 CUDA Inference - forward + sampling only, no backward/optimizer.
*/

// TESTING skips main() in train_gpt2.cu so we can define our own
#define TESTING
#include "train_gpt2.cu"

// parse space-separated token ids from a string into buf, return count
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

int main(int argc, char *argv[]) {
    const char* load_filename  = "gpt2_124M_bf16.bin";
    const char* prompt_tokens  = NULL;
    int genT                   = 256;
    int stop_at_eot            = 0;   // -x 1 to stop at <|endoftext|>
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
    model.seq_len = T;
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

    int eot_token   = tokenizer.eot_token;
    int prompt_len  = 0;

    // fill with eot, then overwrite with prompt tokens if provided
    for (int i = 0; i < B * T; i++) { gen_tokens[i] = eot_token; }

    if (prompt_tokens != NULL) {
        prompt_len = parse_tokens(prompt_tokens, gen_tokens, T);
        if (prompt_len <= 0) {
            fprintf(stderr, "failed to parse prompt tokens\n");
            return 1;
        }
        // prefill: run forward over the full prompt at once
        printf("prompt: ");
        for (int i = 0; i < prompt_len; i++) {
            if (tokenizer.init_ok) {
                safe_printf(tokenizer_decode(&tokenizer, gen_tokens[i]));
            } else {
                printf("%d ", gen_tokens[i]);
            }
        }
        printf("\n---\n");
        gpt2_forward(&model, gen_tokens, B, prompt_len);
    }

    unsigned long long rng_state = seed;

    printf("generating:\n---\n");
    // if we had a prompt, start generating from prompt_len
    // the first new token uses logits from position prompt_len-1
    int t_start = (prompt_len > 0) ? prompt_len : 1;

    for (int t = t_start; t < genT; t++) {
        // for t == t_start after prefill, logits are already ready at prompt_len-1
        // for subsequent steps, run forward over [0..t-1]
        if (t > t_start || prompt_len == 0) {
            gpt2_forward(&model, gen_tokens, B, CEIL_DIV(t, min(T, 256)) * min(T, 256));
        }

        // logits for the last real token
        int logit_pos = (t == t_start && prompt_len > 0) ? prompt_len - 1 : t - 1;
        floatX* logits = model.acts.output + logit_pos * model.config.padded_vocab_size;
        cudaCheck(cudaMemcpy(cpu_logits_raw, logits,
                             model.config.vocab_size * sizeof(floatX),
                             cudaMemcpyDeviceToHost));
        for (int i = 0; i < model.config.vocab_size; i++) {
            cpu_logits[i] = (float)cpu_logits_raw[i];
        }

        float coin = random_f32(&rng_state);
        int next_token = sample_softmax(cpu_logits, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;

        if (next_token == eot_token) {
            if (stop_at_eot) { break; }
            printf("<|endoftext|>");
            fflush(stdout);
            continue;
        }

        if (tokenizer.init_ok) {
            safe_printf(tokenizer_decode(&tokenizer, next_token));
        } else {
            printf("%d ", next_token);
        }
        fflush(stdout);
    }
    printf("\n---\n");

    free(gen_tokens);
    free(cpu_logits_raw);
    free(cpu_logits);
    tokenizer_free(&tokenizer);
    gpt2_free(&model);
    common_free(model);
    return 0;
}
