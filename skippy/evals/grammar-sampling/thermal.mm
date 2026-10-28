#import <Foundation/Foundation.h>

int thermal_state() {
    return static_cast<int>([[NSProcessInfo processInfo] thermalState]);
}
