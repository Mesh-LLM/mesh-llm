#import <Foundation/Foundation.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *mesh_llm_gpu_bench_metal_json(char **error_out);
void mesh_llm_gpu_bench_free(void *ptr);

int main(int argc, char **argv) {
    int json_mode = 0;
    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--json") == 0) {
            json_mode = 1;
        }
    }

    char *error = NULL;
    char *json = mesh_llm_gpu_bench_metal_json(&error);
    if (json == NULL) {
        if (json_mode) {
            printf("{\"error\":\"%s\"}\n", error == NULL ? "Metal benchmark failed" : error);
        } else {
            fprintf(stderr, "%s\n", error == NULL ? "Metal benchmark failed" : error);
        }
        mesh_llm_gpu_bench_free(error);
        return 1;
    }

    printf("%s\n", json);
    mesh_llm_gpu_bench_free(json);
    return 0;
}
