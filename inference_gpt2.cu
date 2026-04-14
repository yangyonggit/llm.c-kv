/*
GPT-2 CUDA Inference - forward + sampling only, no backward/optimizer.
*/

// TESTING skips main() in train_gpt2.cu so we can define our own
#define TESTING
#include "train_gpt2.cu"

int main(int argc, char *argv[]) {
    const char* load_filename = "gpt2_124M_bf16.bin";
    int genT = 256;

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) { fprintf(stderr, "missing arg\n"); return 1; }
        if (argv[i][1] == 'e') { load_filename = argv[i+1]; }
        else if (argv[i][1] == 'g') { genT = atoi(argv[i+1]); }
    }

    multi_gpu_config = multi_gpu_config_init(1, 0, 1, (char*)"", (char*)"", (char*)"fs");
    common_start(false, false);

    GPT2 model;
    gpt2_init_common(&model);
    gpt2_build_from_checkpoint(&model, load_filename);

    // allocate activations only (no gradients, no optimizer state)
    int B = 1, T = 1024;
    model.batch_size = B;
    model.seq_len = T;
    fill_in_activation_sizes(&model.acts, model.acts_specs, B, T, model.config, model.recompute);
    model.acts_memory = malloc_and_point_activations(model.acts_specs);
    cudaCheck(cudaMalloc((void**)&model.inputs, B * T * sizeof(int)));
    cudaCheck(cudaMalloc((void**)&model.targets, B * T * sizeof(int)));
    cudaCheck(cudaMallocHost((void**)&model.cpu_losses, B * T * sizeof(float)));

    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    int*    gen_tokens     = (int*)   mallocCheck(B * T * sizeof(int));
    floatX* cpu_logits_raw = (floatX*)mallocCheck(model.config.vocab_size * sizeof(floatX));
    float*  cpu_logits     = (float*) mallocCheck(model.config.vocab_size * sizeof(float));

    // start with <|endoftext|> token
    int eot_token = tokenizer.eot_token;
    for (int i = 0; i < B * T; i++) { gen_tokens[i] = eot_token; }

    unsigned long long rng_state = 1337;

    printf("generating:\n---\n");
    for (int t = 1; t < genT; t++) {
        // forward (no KV cache yet)
        gpt2_forward(&model, gen_tokens, 1, CEIL_DIV(t, min(T, 256)) * min(T, 256));

        // get logits for last token position
        floatX* logits = model.acts.output + (t - 1) * model.config.padded_vocab_size;
        cudaCheck(cudaMemcpy(cpu_logits_raw, logits,
                             model.config.vocab_size * sizeof(floatX),
                             cudaMemcpyDeviceToHost));
        for (int i = 0; i < model.config.vocab_size; i++) {
            cpu_logits[i] = (float)cpu_logits_raw[i];
        }

        // sample and append
        float coin = random_f32(&rng_state);
        int next_token = sample_softmax(cpu_logits, model.config.vocab_size, coin);
        gen_tokens[t] = next_token;

        if (tokenizer.init_ok) {
            const char* token_str = tokenizer_decode(&tokenizer, next_token);
            safe_printf(token_str);
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
