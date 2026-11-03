// CHECKS: skippy-no-raw-pointer
struct Node {
    int* payload;
    // CHECK: [[@LINE-1]]:10: warning: raw pointer 'payload'
};

int gather(int* data);
// CHECK: [[@LINE-1]]:17: warning: raw pointer 'data'

int* produce();
// CHECK: [[@LINE-1]]:6: warning: raw pointer 'produce'

using Callback = void (*)(int);  // negative: function pointers are not data pointers

int main(int argc, char** argv) { return argc + (argv != nullptr); }  // negative: argv is allow-listed
