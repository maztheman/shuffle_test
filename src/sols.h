#pragma once
#include <cstdint>

struct	sols_s;
typedef sols_s sols_t;
struct candidate_s;
typedef candidate_s candidate_t;

void compress(uint8_t *out, uint32_t *inputs, uint32_t n);

uint32_t verify_sol(sols_t *sols, unsigned sol_i);
uint32_t verify_sol(candidate_t *sols, unsigned sol_i, uint8_t (&valid)[16]);
