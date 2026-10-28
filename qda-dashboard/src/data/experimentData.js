// Extracted from: groups/group1_qda/outputs/qda_80_20_experiment/run_20261015T085434Z/

export const overallMetrics = {
  accuracy: 0.9047,
  balancedAccuracy: 0.9047,
  macroPrecision: 0.8903,
  macroRecall: 0.9047,
  macroF1: 0.8949,
  weightedPrecision: 0.9085,
  weightedRecall: 0.9047,
  weightedF1: 0.9043,
  trainAccuracy: 0.9193,
  trainMacroF1: 0.9120,
};

export const perClassMetrics = [
  { class: "Adware",      precision: 0.8786, recall: 0.9801, f1: 0.9266, support: 251 },
  { class: "Banking",     precision: 0.8119, recall: 0.8738, f1: 0.8417, support: 420 },
  { class: "Benign",      precision: 0.8440, recall: 0.9192, f1: 0.8800, support: 359 },
  { class: "Riskware",    precision: 0.9338, recall: 0.7760, f1: 0.8476, support: 509 },
  { class: "SMS Malware", precision: 0.9832, recall: 0.9744, f1: 0.9788, support: 781 },
];

export const classLabels = ["Adware", "Banking", "Benign", "Riskware", "SMS Malware"];

export const confusionMatrix = [
  [246,  1,  1,  3,  0],
  [  9,367, 18, 18,  8],
  [  1, 23,330,  5,  0],
  [ 24, 48, 37,395,  5],
  [  0, 13,  5,  2,761],
];

export const cvResults = [
  { regParam: 0.00, status: "FAIL",  meanF1: null,   stdF1: null,   error: "LinAlgError: SMS Malware covariance not full rank" },
  { regParam: 0.10, status: "PASS",  meanF1: 0.9076, stdF1: 0.0053, folds: [0.9169, 0.9018, 0.9085, 0.9035, 0.9073] },
  { regParam: 0.25, status: "PASS",  meanF1: 0.8797, stdF1: 0.0064, folds: [0.8831, 0.8683, 0.8865, 0.8775, 0.8832] },
  { regParam: 0.50, status: "PASS",  meanF1: 0.8622, stdF1: 0.0108, folds: [0.8707, 0.8472, 0.8709, 0.8511, 0.8713] },
  { regParam: 0.75, status: "PASS",  meanF1: 0.8456, stdF1: 0.0115, folds: [0.8518, 0.8340, 0.8545, 0.8297, 0.8579] },
  { regParam: 1.00, status: "PASS",  meanF1: 0.7096, stdF1: 0.0075, folds: [0.7217, 0.7069, 0.7098, 0.6984, 0.7114] },
];

export const trainDistribution = {
  Adware: 1002,
  Banking: 1680,
  Benign: 1436,
  Riskware: 2037,
  "SMS Malware": 3123,
};

export const testDistribution = {
  Adware: 251,
  Banking: 420,
  Benign: 359,
  Riskware: 509,
  "SMS Malware": 781,
};

export const selectedFeatures = [
  "ANDROID.permission.VIBRATE",
  "ContentResolver:insert",
  "PC",
  "ASSET",
  "WB_EXP",
  "Web",
  "WA",
  "num_permissions.1",
  "PROVIDER",
  "de.aktiwir.aktibmi.util.Reminder",
  "HAP_BIND",
  "main_activity.1",
  "NONA",
  "remount",
  "TTComp",
  "android.intent.action.PACKAGE_REMOVED.1",
  "DOME_TOGGLE",
  "android.intent.action.PACKAGE_REPLACED.1",
  "SmsManager:sendTextMessage",
  "FoxPro",
  "FilterService.ACTION_FILTER_NO_LICENSE",
  "WIDGET_UPDATE_ACTION_TO_NEXT_ARTICLE",
  "android.intent.action.PACKAGE_ADDED.1",
  "ensufwosr.rokcvyfjtdp.sdjhwvlha",
  "Lcom/asdjklqw/echovox/MainA;=>onCreate",
  "MIPSEL-BE",
  "airplane_mode settings command",
  "Context:getApplicationInfo",
  "android.intent.action.BOOT_COMPLETED.1",
  "GARMIN_PRIVACY_POLICY",
  "num_third_part_permissions.1",
  "UTIL",
  "JAVA_REFLECTION",
  "Convex",
  "CoreAudio",
  "INPUTMETHODSERVICE",
  "num_services.1",
  "JPEG",
  "APP_EULA",
  "VAX",
  "android.intent.action.WEB_SEARCH.1",
  "La/a",
  "DAILY_CALENDAR",
  "TIMER",
  "RAR",
  "JAVA_REFLECTION.1",
  "Lcom/asdjklqw/echovox/MainA$3;=>onClick",
  "android.intent.action.BATTERY_CHANGED_ACTION.1",
  "UPDATE_BACKGROUND_AUDIO_NOTIFICATION",
  "HTML",
];

export const experimentMetadata = {
  experimentType: "80:20 Stratified Split",
  timestamp: "2026-10-15T08:54:34Z",
  dataset: "MalDroid-2020",
  totalSamples: 11598,
  trainSamples: 9278,
  testSamples: 2320,
  featureModality: "Static",
  featureSelectionMethod: "Mutual Information",
  featureCount: 50,
  originalFeatures: 50621,
  selectedRegParam: 0.1,
  randomSeed: 42,
  splitStrategy: "Stratified 80:20",
  packageVersions: {
    python: "3.13.7",
    "scikit-learn": "1.8.0",
    numpy: "2.4.3",
    pandas: "3.0.1",
  },
};
