// Native app entry point. All paths are relative to this bundle, never the
// builder's machine. Embed Python so macOS keeps SIGMA's bundle identity.
#import <Cocoa/Cocoa.h>
#include <Python.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        NSString *resources = [[NSBundle mainBundle] resourcePath];
        NSString *runtime = [resources stringByAppendingPathComponent:@"runtime"];
        NSString *python = [runtime stringByAppendingPathComponent:@"bin/python3"];
        NSString *script = [runtime stringByAppendingPathComponent:@"sigma-desktop/launch.py"];
        if (![[NSFileManager defaultManager] fileExistsAtPath:script]) {
            NSAlert *alert = [[NSAlert alloc] init];
            alert.messageText = @"SIGMA could not start";
            alert.informativeText = @"The application is incomplete. Copy the complete SIGMA.app from the disk image.";
            [alert runModal];
            return 1;
        }
        PyConfig config;
        PyConfig_InitIsolatedConfig(&config);
        config.write_bytecode = 0;  // Never modify a signed or read-only app.
        PyStatus status;
#define CHECK(call) do { status = (call); if (PyStatus_Exception(status)) goto failed; } while (0)
        CHECK(PyConfig_SetBytesString(&config, &config.home, runtime.fileSystemRepresentation));
        // Child processes use the bundled interpreter, not the GUI launcher.
        CHECK(PyConfig_SetBytesString(&config, &config.executable, python.fileSystemRepresentation));
        CHECK(PyConfig_SetBytesString(&config, &config.program_name, argv[0]));
        CHECK(PyConfig_SetBytesString(&config, &config.run_filename, script.fileSystemRepresentation));
        argv[0] = (char *)script.fileSystemRepresentation;
        CHECK(PyConfig_SetBytesArgv(&config, argc, argv));
        CHECK(Py_InitializeFromConfig(&config));
        PyConfig_Clear(&config);
        return Py_RunMain();
failed:
        {
            NSAlert *alert = [[NSAlert alloc] init];
            alert.messageText = @"SIGMA could not initialize Python";
            alert.informativeText = status.err_msg ? [NSString stringWithUTF8String:status.err_msg] : @"Reinstall the complete application.";
            PyConfig_Clear(&config);
            [alert runModal];
            return 1;
        }
    }
}
