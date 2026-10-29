#import <Foundation/Foundation.h>

int thermal_state(void) {
    return (int)[[NSProcessInfo processInfo] thermalState];
}
