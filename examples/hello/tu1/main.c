/* proprietary application source: never leaves the vendor */
#include "secret.h"
int main(void) { return secret_compute(42) == 0x5eed ? 0 : 1; }
