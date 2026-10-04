// A small program for the attested compiler: iterative fibonacci + a struct.
struct pair { int a; int b; };

int fib(int n) {
  struct pair p;
  p.a = 0; p.b = 1;
  for (int i = 0; i < n; i++) {
    int t = p.a + p.b;
    p.a = p.b;
    p.b = t;
  }
  return p.a;
}

int main() {
  return fib(10) == 55 ? 0 : 1;
}
