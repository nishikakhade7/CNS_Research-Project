"""Reference name lists used by Stage 03 to classify KronoDroid's 484 columns into
dynamic (syscall), static (Android permission / APK-structural), and metadata groups
by NAME-MATCHING, not by column position.

Sources (declared, not invented):
  - Syscall names: the ARM EABI 32-bit Linux syscall table (asm-generic/unistd.h style
    names with the `32`/`64` ARM EABI suffixes, e.g. getuid32, mmap2, _llseek,
    ftruncate64) — this is the standard Android/Bionic libc syscall naming convention.
  - Permission names: the public `android.Manifest.permission` constant names from the
    Android SDK (e.g. ACCESS_FINE_LOCATION, INTERNET, CAMERA, BIND_*_SERVICE).

These lists are necessarily a reconstruction from documented knowledge of the Android/
Linux API surface, not a machine-readable dump of a specific AOSP source file checked
into this repo. Any column that matches neither list is NOT silently assigned — it is
left in `unclassified`, reported explicitly, and must be manually reviewed (see Stage 03
output `unclassified_columns`).
"""

from __future__ import annotations

KNOWN_SYSCALL_NAMES: set[str] = {
    "execve", "getuid32", "getgid32", "geteuid32", "getegid32", "getresuid32", "getresgid32",
    "readahead", "getgroups32", "getpgid", "getppid", "getsid", "setsid", "setgid32", "setuid32",
    "setreuid32", "setresuid32", "setresgid32", "brk", "kill", "tgkill", "ptrace", "getrusage",
    "getpriority", "setpriority", "ugetrlimit", "setrlimit", "prlimit64", "setgroups32", "setpgid",
    "setregid32", "chroot", "prctl", "capget", "capset", "sigaltstack", "acct", "read", "write",
    "pread64", "pwrite64", "preadv", "pwritev", "close", "getpid", "munmap", "mremap", "msync",
    "mprotect", "madvise", "mlock", "munlock", "mlockall", "munlockall", "mincore", "ioctl",
    "readv", "writev", "fcntl64", "flock", "fchmod", "dup", "pipe2", "dup3", "fsync", "fdatasync",
    "fchown32", "sync", "fsetxattr", "fgetxattr", "flistxattr", "fremovexattr", "getdents64",
    "openat", "faccessat", "fchmodat", "fchownat", "fstatat64", "linkat", "mkdirat", "mknodat",
    "readlinkat", "renameat", "symlinkat", "unlinkat", "utimensat", "lseek", "_llseek",
    "ftruncate64", "sendfile", "sendfile64", "truncate", "truncate64", "mmap2", "fallocate",
    "fadvise64_64", "fstatfs64", "statfs64", "fstat64", "chdir", "mount", "umount2", "getcwd",
    "fchdir", "setxattr", "lsetxattr", "getxattr", "lgetxattr", "listxattr", "llistxattr",
    "removexattr", "lremovexattr", "swapon", "swapoff", "settimeofday", "times", "nanosleep",
    "clock_settime", "clock_getres", "clock_nanosleep", "getitimer", "setitimer", "timer_create",
    "timer_settime", "timer_gettime", "timer_getoverrun", "timer_delete", "timerfd_create",
    "timerfd_settime", "timerfd_gettime", "adjtimex", "clock_adjtime", "sigaction", "rt_sigaction",
    "rt_sigpending", "rt_sigprocmask", "rt_sigsuspend", "rt_sigtimedwait", "rt_sigqueueinfo",
    "signalfd4", "socket", "bind", "connect", "listen", "getsockname", "getpeername", "socketpair",
    "sendto", "recvfrom", "shutdown", "setsockopt", "getsockopt", "sendmsg", "recvmsg", "accept4",
    "recvmmsg", "sendmmsg", "sched_setscheduler", "sched_getscheduler", "sched_yield",
    "sched_setparam", "sched_getparam", "sched_get_priority_max", "sched_get_priority_min",
    "sched_rr_get_interval", "sched_setaffinity", "setns", "unshare", "sched_getaffinity",
    "getcpu", "uname", "umask", "reboot", "init_module", "delete_module", "syslog", "sysinfo",
    "personality", "tee", "splice", "vmsplice", "epoll_create1", "epoll_ctl", "epoll_pwait",
    "eventfd2", "exit_group", "exit", "inotify_init1", "inotify_add_watch", "inotify_rm_watch",
    "pselect6", "ppoll", "process_vm_readv", "set_tid_address", "setfsgid", "setfsuid",
    "sethostname", "wait4", "waitid", "set_thread_area", "clock_gettime", "gettimeofday", "clone",
    "futex", "vfork", "rt_sigreturn", "restart_syscall", "getrandom", "gettid", "epoll_wait",
    "stat64", "pipe", "pread", "getrlimit", "pwrite", "recv", "open",
}

# Non-syscall technical columns that sit in the same block of the file for a documented
# reason: unassigned/reserved Linux syscall-table slots (SYS_300..SYS_369, the ARM EABI
# range later repurposed/left unused across kernel versions) plus one vendor-specific
# raw syscall number (syscall_983042) and the block's own summary column (nr_syscalls).
# Treated as "dynamic" (same measurement axis as the syscalls) but flagged separately
# since they are not resolvable libc syscall *names*.
SYSCALL_RESERVED_OR_SUMMARY: set[str] = {f"SYS_{n}" for n in range(300, 370)} | {
    "syscall_983042", "nr_syscalls",
}

KNOWN_ANDROID_PERMISSIONS: set[str] = {
    "ACCEPT_HANDOVER", "ACCESS_BACKGROUND_LOCATION", "ACCESS_CHECKIN_PROPERTIES",
    "ACCESS_COARSE_LOCATION", "ACCESS_FINE_LOCATION", "ACCESS_LOCATION_EXTRA_COMMANDS",
    "ACCESS_MEDIA_LOCATION", "ACCESS_NETWORK_STATE", "ACCESS_NOTIFICATION_POLICY",
    "ACCESS_WIFI_STATE", "ACCOUNT_MANAGER", "ACTIVITY_RECOGNITION", "ADD_VOICEMAIL",
    "ANSWER_PHONE_CALLS", "BATTERY_STATS", "BIND_ACCESSIBILITY_SERVICE", "BIND_APPWIDGET",
    "BIND_AUTOFILL_SERVICE", "BIND_CALL_REDIRECTION_SERVICE",
    "BIND_CARRIER_MESSAGING_CLIENT_SERVICE", "BIND_CARRIER_MESSAGING_SERVICE",
    "BIND_CARRIER_SERVICES", "BIND_CHOOSER_TARGET_SERVICE", "BIND_CONDITION_PROVIDER_SERVICE",
    "BIND_CONTROLS", "BIND_DEVICE_ADMIN", "BIND_DREAM_SERVICE", "BIND_INCALL_SERVICE",
    "BIND_INPUT_METHOD", "BIND_MIDI_DEVICE_SERVICE", "BIND_NFC_SERVICE",
    "BIND_NOTIFICATION_LISTENER_SERVICE", "BIND_PRINT_SERVICE",
    "BIND_QUICK_ACCESS_WALLET_SERVICE", "BIND_QUICK_SETTINGS_TILE", "BIND_REMOTEVIEWS",
    "BIND_SCREENING_SERVICE", "BIND_TELECOM_CONNECTION_SERVICE", "BIND_TEXT_SERVICE",
    "BIND_TV_INPUT", "BIND_VISUAL_VOICEMAIL_SERVICE", "BIND_VOICE_INTERACTION",
    "BIND_VPN_SERVICE", "BIND_VR_LISTENER_SERVICE", "BIND_WALLPAPER", "BLUETOOTH",
    "BLUETOOTH_ADMIN", "BLUETOOTH_PRIVILEGED", "BODY_SENSORS", "BROADCAST_PACKAGE_REMOVED",
    "BROADCAST_SMS", "BROADCAST_STICKY", "BROADCAST_WAP_PUSH", "CALL_COMPANION_APP",
    "CALL_PHONE", "CALL_PRIVILEGED", "CAMERA", "CAPTURE_AUDIO_OUTPUT",
    "CHANGE_COMPONENT_ENABLED_STATE", "CHANGE_CONFIGURATION", "CHANGE_NETWORK_STATE",
    "CHANGE_WIFI_MULTICAST_STATE", "CHANGE_WIFI_STATE", "CLEAR_APP_CACHE",
    "CONTROL_LOCATION_UPDATES", "DELETE_CACHE_FILES", "DELETE_PACKAGES", "DIAGNOSTIC",
    "DISABLE_KEYGUARD", "DUMP", "EXPAND_STATUS_BAR", "FACTORY_TEST", "FOREGROUND_SERVICE",
    "GET_ACCOUNTS", "GET_ACCOUNTS_PRIVILEGED", "GET_PACKAGE_SIZE", "GET_TASKS", "GLOBAL_SEARCH",
    "INSTALL_LOCATION_PROVIDER", "INSTALL_PACKAGES", "INSTALL_SHORTCUT",
    "INSTANT_APP_FOREGROUND_SERVICE", "INTERACT_ACROSS_PROFILES", "INTERNET",
    "KILL_BACKGROUND_PROCESSES", "LOADER_USAGE_STATS", "LOCATION_HARDWARE", "MANAGE_DOCUMENTS",
    "MANAGE_EXTERNAL_STORAGE", "MANAGE_OWN_CALLS", "MASTER_CLEAR", "MEDIA_CONTENT_CONTROL",
    "MODIFY_AUDIO_SETTINGS", "MODIFY_PHONE_STATE", "MOUNT_FORMAT_FILESYSTEMS",
    "MOUNT_UNMOUNT_FILESYSTEMS", "NFC", "NFC_PREFERRED_PAYMENT_INFO", "NFC_TRANSACTION_EVENT",
    "PACKAGE_USAGE_STATS", "PERSISTENT_ACTIVITY", "PROCESS_OUTGOING_CALLS",
    "QUERY_ALL_PACKAGES", "READ_CALENDAR", "READ_CALL_LOG", "READ_CONTACTS",
    "READ_EXTERNAL_STORAGE", "READ_INPUT_STATE", "READ_LOGS", "READ_PHONE_NUMBERS",
    "READ_PHONE_STATE", "READ_PRECISE_PHONE_STATE", "READ_SMS", "READ_SYNC_SETTINGS",
    "READ_SYNC_STATS", "READ_VOICEMAIL", "REBOOT", "RECEIVE_BOOT_COMPLETED", "RECEIVE_MMS",
    "RECEIVE_SMS", "RECEIVE_WAP_PUSH", "RECORD_AUDIO", "REORDER_TASKS",
    "REQUEST_COMPANION_RUN_IN_BACKGROUND", "REQUEST_COMPANION_USE_DATA_IN_BACKGROUND",
    "REQUEST_DELETE_PACKAGES", "REQUEST_IGNORE_BATTERY_OPTIMIZATIONS",
    "REQUEST_INSTALL_PACKAGES", "REQUEST_PASSWORD_COMPLEXITY", "RESTART_PACKAGES",
    "SEND_RESPOND_VIA_MESSAGE", "SEND_SMS", "SET_ALARM", "SET_ALWAYS_FINISH",
    "SET_ANIMATION_SCALE", "SET_DEBUG_APP", "SET_PREFERRED_APPLICATIONS", "SET_PROCESS_LIMIT",
    "SET_TIME", "SET_TIME_ZONE", "SET_WALLPAPER", "SET_WALLPAPER_HINTS",
    "SIGNAL_PERSISTENT_PROCESSES", "SMS_FINANCIAL_TRANSACTIONS", "START_VIEW_PERMISSION_USAGE",
    "STATUS_BAR", "SYSTEM_ALERT_WINDOW", "TRANSMIT_IR", "UNINSTALL_SHORTCUT",
    "UPDATE_DEVICE_STATS", "USE_BIOMETRIC", "USE_FINGERPRINT", "USE_FULL_SCREEN_INTENT",
    "USE_SIP", "VIBRATE", "WAKE_LOCK", "WRITE_APN_SETTINGS", "WRITE_CALENDAR",
    "WRITE_CALL_LOG", "WRITE_CONTACTS", "WRITE_EXTERNAL_STORAGE", "WRITE_GSERVICES",
    "WRITE_SECURE_SETTINGS", "WRITE_SETTINGS", "WRITE_SYNC_SETTINGS", "WRITE_VOICEMAIL",
}

# Derived summary columns over the permission block (not individual permission flags).
PERMISSION_SUMMARY_COLS: set[str] = {
    "nr_permissions", "normal", "dangerous", "signature", "custom_yes", "nr_custom", "total_perm",
}

# APK-structural static features (manifest/component counts, file size) — static because
# they come from the APK's static manifest/zip structure, not runtime behavior.
APK_STRUCTURAL_COLS: set[str] = {
    "CFileSize", "UFileSize", "FilesInsideAPK", "Activities", "NrIntServices",
    "NrIntServicesActions", "NrIntActivities", "NrIntActivitiesActions", "NrIntReceivers",
    "NrIntReceiversActions", "TotalIntentFilters", "NrServices",
}

# NrContactedIps is a genuine dynamic/behavioral measurement (network contacts observed
# during execution), not static or metadata — kept separate from the syscall block only
# because it isn't a named syscall.
DYNAMIC_BEHAVIORAL_COLS: set[str] = {"NrContactedIps"}

# Pure identifiers / provenance metadata — never fed to the model as features.
METADATA_COLS: set[str] = {
    "Package", "Malware", "sha256", "EarliestModDate", "HighestModDate", "TimesSubmitted",
    "Scanners", "Detection_Ratio", "MalFamily",
}
