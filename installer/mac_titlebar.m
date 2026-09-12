// Keep native controls/dragging and the accessible NSWindow title. macOS 26
// places titles at the left even with toolbarStyle=Expanded and no toolbar,
// so center a passive title field in the existing native title-bar view.
#import <Cocoa/Cocoa.h>
#import <objc/runtime.h>

@interface SIGMATitleField : NSTextField
@end
@implementation SIGMATitleField
- (NSView *)hitTest:(NSPoint)point { return nil; }
@end

@interface SIGMATitleController : NSObject
@property(weak) NSWindow *window;
@property(strong) SIGMATitleField *label;
@property(strong) NSArray<NSLayoutConstraint *> *constraints;
- (void)update;
@end

@implementation SIGMATitleController
- (void)update {
    NSWindow *window = self.window;
    NSButton *button = [window standardWindowButton:NSWindowCloseButton];
    NSView *host = button.superview;
    if (!host) return;
    // Use only public view relationships, never private titlebar classes.
    while (host.superview && NSWidth(host.bounds) < NSWidth(window.frame) - 2)
        host = host.superview;
    window.titleVisibility = NSWindowTitleHidden;
    self.label.stringValue = window.title;
    self.label.toolTip = window.title;
    if (self.label.superview == host) {
        [host layoutSubtreeIfNeeded];
        return;
    }
    [NSLayoutConstraint deactivateConstraints:self.constraints ?: @[]];
    [self.label removeFromSuperview];
    [host addSubview:self.label];
    self.constraints = @[
        [self.label.centerXAnchor constraintEqualToAnchor:host.centerXAnchor],
        [self.label.centerYAnchor constraintEqualToAnchor:button.centerYAnchor],
        [self.label.leadingAnchor constraintGreaterThanOrEqualToAnchor:host.leadingAnchor constant:100],
        [self.label.trailingAnchor constraintLessThanOrEqualToAnchor:host.trailingAnchor constant:-100]
    ];
    [NSLayoutConstraint activateConstraints:self.constraints];
    // Qt's processEvents does not guarantee an AppKit layout/display pass.
    // A cold launch can otherwise retain the label's initial x=0 frame until
    // the first normal native event-loop turn. Resolve constraints now, also
    // when reinstalling the label after a fullscreen/titlebar transition.
    [host layoutSubtreeIfNeeded];
}
- (void)windowChanged:(NSNotification *)notification { [self update]; }
- (void)dealloc { [[NSNotificationCenter defaultCenter] removeObserver:self]; }
@end

static char titleControllerKey;

int SIGMACenterWindowTitle(void *viewPointer) {
    if (![NSThread isMainThread] || !viewPointer) return 0;
    NSView *view = (__bridge NSView *)viewPointer;
    NSWindow *window = view.window;
    if (!window) return 0;
    SIGMATitleController *controller = objc_getAssociatedObject(window, &titleControllerKey);
    if (!controller) {
        controller = [SIGMATitleController new];
        controller.window = window;
        controller.label = [SIGMATitleField labelWithString:window.title];
        controller.label.font = [NSFont systemFontOfSize:13 weight:NSFontWeightSemibold];
        controller.label.textColor = NSColor.labelColor;
        controller.label.alignment = NSTextAlignmentCenter;
        controller.label.lineBreakMode = NSLineBreakByTruncatingTail;
        controller.label.translatesAutoresizingMaskIntoConstraints = NO;
        objc_setAssociatedObject(window, &titleControllerKey, controller, OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        for (NSNotificationName name in @[NSWindowDidResizeNotification,
                NSWindowDidEnterFullScreenNotification, NSWindowDidExitFullScreenNotification])
            [[NSNotificationCenter defaultCenter] addObserver:controller selector:@selector(windowChanged:)
                                                        name:name object:window];
    }
    [controller update];
    return controller.label.superview != nil;
}

// Read-only geometry probe used by native GUI regression tests. Returns NaN
// if AppKit's title field is unavailable (for example while in fullscreen).
static NSTextField *titleField(NSView *view, NSString *title) {
    if ([view isKindOfClass:[SIGMATitleField class]] && !view.hiddenOrHasHiddenAncestor &&
        [((NSTextField *)view).stringValue isEqualToString:title]) {
        return (NSTextField *)view;
    }
    for (NSView *child in view.subviews) {
        NSTextField *found = titleField(child, title);
        if (found) return found;
    }
    return nil;
}

double SIGMAWindowTitleOffset(void *viewPointer) {
    if (![NSThread isMainThread] || !viewPointer) return NAN;
    NSWindow *window = ((__bridge NSView *)viewPointer).window;
    NSTextField *field = titleField(window.contentView.superview, window.title);
    if (!field || field.hidden) return NAN;
    NSRect rect = [field convertRect:field.bounds toView:nil];
    return NSMidX(rect) - NSWidth(window.frame) / 2.0;
}
