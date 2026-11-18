/*
 * qvcv_adaptive.js
 *
 * Client-side psychometric, scheduling, and clinical-scoring engine for qVCV:
 *   1. Non-repeating balanced consonant + alternating token scheduler (both Standard
 *      Fixed-SNR 5x20 blocks and Adaptive-SNR 10-consonant blocks).
 *   2. Audiogram-blind Hierarchical Bayesian Adaptive-SNR tracker (1-dB resolution,
 *      -18..+30 dB SNR; 3-point Gauss-Hermite expected Fisher information + 2-step
 *      Newton-Raphson MAP & exact 10x10 posterior covariance).
 *   3. Post-test hierarchical MAP behavioral SNR-loss decoder, optional Plomp (1978)
 *      audibility vs. suprathreshold coding decomposition (when audiogram is supplied),
 *      exact published PPTA multinomial logistic regression (Hajicek, Harris, & Neely, 2026),
 *      and Miller-Nicely Voicing / Manner / Place information transmission.
 *
 * Auto-calibrated to the 98-listener qtest_dataset cohort (32 NH, 66 HL).
 * Runs 100% locally in browser or Node.js with zero external dependencies.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.QVCVEngine = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const STIM_CONSONANTS = ["B", "D", "G", "K", "N", "S", "SH", "T", "V", "Z"];
  const DISP_CONSONANTS = ["B", "D", "G", "K", "N", "S", "SH", "T", "V", "Z", "???"];
  const AUD_FREQS_HZ = [250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000];

  /* Consonant primary audiometric cue-band indices into AUD_FREQS_HZ */
  const CUE_BAND_IDX = [
    [1, 2, 3],       /* B : 0.5, 0.75, 1.0 kHz */
    [5, 6, 7],       /* D : 2.0, 3.0, 4.0 kHz */
    [3, 4, 5],       /* G : 1.0, 1.5, 2.0 kHz */
    [4, 7, 8],       /* K : 1.5, 4.0, 6.0 kHz */
    [0, 1, 3, 4],    /* N : 0.25, 0.5, 1.0, 1.5 kHz */
    [7, 8, 9],       /* S : 4.0, 6.0, 8.0 kHz */
    [5, 6, 7],       /* SH: 2.0, 3.0, 4.0 kHz */
    [7, 8, 9],       /* T : 4.0, 6.0, 8.0 kHz */
    [2, 4, 5, 7],    /* V : 0.75, 1.5, 2.0, 4.0 kHz */
    [6, 7, 8, 9]     /* Z : 3.0, 4.0, 6.0, 8.0 kHz */
  ];

  /* Phonetic feature classes for Miller-Nicely information transmission:
   * Voicing: 1=Voiced (B,D,G,N,V,Z), 0=Voiceless (K,S,SH,T)
   * Manner : 0=Stop (B,D,G,K,T), 1=Fricative (S,SH,V,Z), 2=Nasal (N)
   * Place  : 0=Labial (B,V), 1=Alveolar (D,N,S,T,Z), 2=Palatal/Velar (G,K,SH)
   */
  const FEAT_VOICING = [1, 1, 1, 0, 1, 0, 0, 0, 1, 1];
  const FEAT_MANNER  = [0, 0, 0, 0, 2, 1, 1, 0, 1, 1];
  const FEAT_PLACE   = [0, 1, 2, 2, 1, 1, 2, 1, 0, 1];

  /* Calibrated 98-listener cohort psychometric constants */
  const CAL = {
    slope: 0.3356,
    gamma: 1.0 / 11.0,
    ceilP: 0.996,
    span: 0.996 - (1.0 / 11.0),
    betaAudiogram: 0.7453,
    ltassDeltaMDb: 8.1674,
    alpha0: [-2.2843, 1.6205, 1.3718, 3.077, 0.4105, 3.3699, 3.7219, 4.8561, -1.61, 4.9146],
    tokHalfDiff: [-0.03883, 0.07915, -0.45335, 0.45201, 0.04978, 0.16521, -0.04749, -0.54923, 0.1824, -0.19625],
    ltass0SnrBandHL: [37.49, 29.51, 34.78, 27.83, 34.74, 23.2, 29.51, 23.2, 33.07, 24.61],
    refSnrDb: [18.0, 4.0, 3.0, 0.0, 15.0, 0.0, 0.0, 0.0, 18.0, 0.0],
    /* Online audiogram-blind hierarchical tracker prior (theta_c = G + delta_c) */
    onlineMuTheta: 4.0,
    onlineSdG: 6.5,
    onlineSdDelta: 4.5,
    /* Post-test Plomp suprathreshold deficit prior (D_c = G + delta_c) */
    supraPriorMuD: 2.382743,
    supraPriorSdG: 2.890568,
    supraPriorSdDelta: 3.707128,
    /* 32-listener Normal-Hearing reference norm (mean & SD in dB) */
    nhMeanShiftDb: -0.300,
    nhSdShiftDb: 1.609,
    nhMeanSupraDb: -0.303563,
    nhSdSupraDb: 1.603045,
    /* Reference 0-dB-SNR LTASS noise RMS matching Stimuli/ltass_noise.wav */
    refNoise0SnrRms: 0.0101796
  };
  CAL.tau0 = CAL.alpha0.map(a => -a / CAL.slope);

  /* Candidate 1-dB SNR bank: -18..+30 dB (49 levels) */
  const SNR_BANK_1DB = Array.from({ length: 49 }, (_, i) => -18 + i);

  /* Exact published mnr_vcv_pta model (Hajicek, Harris, & Neely, 2026) */
  const PPTA_MODEL = {
    B: [[-0.581182, -2.428803, -5.450891, -5.080616, -7.529107, -4.27721, -8.011421, -6.011497, -9.007311, -9.002224, -6.566004, -5.759411, -9.128438, -9.1758, -9.389052], [-0.019047, 0.180461, 0.269943, 0.079389, 0.339684, 0.283618, 0.297029, 0.208354, 0.326672, 0.33786, 0.227873, 0.158188, 0.303839, 0.2814, 0.296086], [-0.19045, 0.211886, 0.063293, 0.144174, 0.018157, -0.001068, 0.043247, 0.073557, -0.07445, -0.020498, -0.031748, -0.062058, -0.025399, -0.029336, -0.061003], [0.195347, 0.074509, 0.186841, 0.195885, 0.016447, -0.007235, 0.046547, -0.028223, -0.011537, 0.076677, 0.069313, 0.058604, 0.040454, 0.118524, 0.066426], [-0.129064, 0.073285, 0.130676, 0.183837, 0.061962, -0.131275, 0.084652, -0.032989, 0.01016, -0.065079, -0.057343, -0.010552, -0.05867, -0.032458, -0.047666], [0.125488, -0.108017, -0.143477, -0.057179, 0.012871, -0.047071, -0.24551, -0.107289, -0.21262, -0.005964, 0.032448, -0.103424, 0.04396, -0.026326, -0.025413], [0.051144, -0.033052, 0.058211, -0.016595, 0.080094, -0.089673, -0.046607, -0.068938, 0.038205, -0.023613, 0.050105, -0.171242, -0.039, -0.039664, -0.049709], [0.078176, 0.030253, -0.107852, -0.129855, -0.002798, -0.000456, -0.054697, 0.190917, 0.028063, 0.003891, -0.060538, 0.028057, 0.052935, -0.037738, 0.011684], [-0.015915, -0.206541, -0.011263, -0.030941, -0.101715, -0.078885, -0.058476, 0.168572, 0.084898, -0.013335, -0.033837, 0.179912, 0.019835, -0.004902, 0.031515], [0.104935, -0.036841, -0.073591, -0.010802, 0.036478, -0.224694, 0.093208, -0.123288, -0.050428, -0.100226, -0.132138, -0.122043, -0.092165, -0.020259, -0.017719], [0.08125, 0.155766, -0.041797, -0.154183, -0.002131, 0.121298, 0.061281, -0.02407, 0.142065, 0.026604, -0.017472, -0.115336, -0.020555, 0.052186, 0.03705]],
    C: [[0.173039, -0.102272, 0.194618, 0.270278, 0.56359, 0.237338, 0.022225, -0.169844, -0.183637, 0.062633], [-0.092607, -0.13619, 0.080764, -0.002969, 0.082827, 0.111511, 0.057483, 0.054256, 0.07919, 0.077638], [-0.021889, -0.021467, 0.001541, -0.046692, -0.023294, 0.002182, 0.023271, -0.105831, -0.041468, -0.02245], [-0.003212, -0.00503, -0.002027, -0.012149, -0.013838, -0.005726, 0.000595, -0.050798, -0.005488, -0.015037], [-0.031669, -0.031075, -0.007635, -0.074327, -0.023238, -0.00761, 0.034417, -0.219768, -0.087987, -0.073275], [-0.003506, 4e-05, -0.010089, -0.017138, 0.004424, -0.003101, 0.017591, -0.002741, 0.040443, 0.005252], [-0.002938, -0.005576, -0.002404, -0.012143, -0.005775, 0.006548, -0.018342, -0.017078, 0.007861, 0.006391], [-0.009183, -0.003892, -0.002247, -0.022853, 0.008162, 0.008977, 0.022398, -0.019038, -0.007776, -0.015694], [-0.009901, -0.019528, 0.008826, -0.02021, -0.002394, -0.011671, 0.040747, -0.069727, 0.061479, -0.04065], [-0.001346, 0.000546, -0.001332, -0.006609, -0.004702, 0.008464, -0.002735, -0.003216, -0.015934, 0.003621], [-0.056206, -0.038409, -0.066193, -0.045175, -0.016157, -0.141046, 0.052696, -0.048787, 0.102237, 0.03629], [0.34265, -0.272492, -0.46228, 0.020884, 0.120333, -0.215162, 0.157062, 0.280516, -0.014892, -0.135963], [-0.035725, -0.09522, -0.048099, -0.039047, 0.088305, 0.056287, -0.120388, -0.025419, 0.256242, 0.124705], [-0.004625, 0.000391, 0.003586, -0.010252, 0.007931, 0.005699, 0.012002, -0.007782, -0.020218, 0.006875], [-0.025302, -0.023858, -0.049776, -0.034976, -0.022796, -0.037426, 0.05461, -0.081659, 0.03543, 0.016361], [-0.005331, 0.003274, -0.008866, -0.015778, -0.002293, 0.02073, 0.000942, 0.014232, -0.006569, 0.006545], [-0.000297, -0.001971, -3.1e-05, -0.001139, 0.001161, 0.000779, -0.008235, -0.002407, 0.007789, 0.005172], [-0.022886, -0.015084, -0.051731, -0.07464, 0.056783, 0.034543, 0.007522, 0.003965, 0.136618, -0.001], [-0.018356, 0.001158, -0.022616, -0.017178, -0.015249, -0.066559, -0.003858, -0.029378, 0.004005, 0.022225], [-0.018444, 0.01606, -0.048587, -0.041031, 0.034266, -0.013093, 0.058466, -0.005791, 0.087339, 0.0494], [-0.04654, 0.046347, -0.04248, -0.050926, -0.022856, -0.23726, -0.097012, 0.006117, -0.031246, 0.035236], [-0.150411, 0.458859, 0.320716, 0.14612, -0.10746, -0.065339, -0.126419, -0.186656, 0.107454, 0.132207], [0.274157, 0.4597, 0.078307, 0.144335, -0.048809, -0.190555, 0.007074, 0.205746, -0.432944, -0.276335], [-0.016956, 0.001335, -0.006588, -0.025712, 0.042, 0.012027, 0.057932, -0.000618, -0.106383, 0.035083], [-0.009315, -0.012336, -0.014406, -0.012092, -0.022536, -0.016734, 0.009826, -0.066794, -0.01166, -0.007922], [-0.01083, 0.005292, -0.029658, -0.039426, 0.022528, 0.016198, 0.031613, 0.01251, 0.007031, 0.016595], [-0.009645, 0.003134, 8.2e-05, -0.009704, 0.004597, -0.020719, -0.003028, -0.024406, -0.018559, -0.01157], [-0.037259, 0.070746, 0.019137, -0.027089, 0.062284, -0.042978, -0.00018, 0.025566, -0.001988, -0.003273], [-0.045772, 0.039443, -0.032248, -0.044647, -0.00997, -0.20067, -0.041745, -0.013446, -0.023432, 0.041687], [-0.077599, 0.212951, -0.083927, -0.129533, 0.170472, -0.134501, 0.035989, 0.169638, -0.089667, 0.306977], [-0.00326, -0.003826, -0.000601, -0.009424, -0.005512, -0.0025, 0.002838, -0.033558, -0.008261, -0.011086], [-0.002178, 0.001965, 0.001594, -0.005824, 0.005164, 0.001499, 0.003101, -0.00594, -0.006863, 0.005491], [-0.005003, -0.005274, -0.001694, -0.017683, 0.000791, -0.003263, 0.001525, -0.044187, -0.02697, -0.030979], [0.343722, 0.195096, 0.090043, -0.235233, 0.198075, -0.236474, -0.108514, -0.093033, 0.471559, -0.172576], [-0.002457, -0.002568, -9.4e-05, -0.006386, -0.002053, -0.001068, 0.002689, -0.020859, -0.006889, -0.007326], [-0.03899, 0.120282, -0.106025, -0.090757, 0.121904, -0.078561, 0.028537, 0.089157, 0.090262, -0.034889], [-0.01436, 0.00935, -0.011591, -0.018141, -0.005994, -0.062674, -0.031689, -0.02227, -0.005678, 0.017732], [-0.039403, 0.097188, 0.018985, -0.056059, 0.107193, -0.031084, 0.030796, 0.078968, 0.086948, 0.045483], [-0.001606, -0.002515, -0.001014, -0.006074, -0.006919, -0.002863, 0.000297, -0.025399, -0.002744, -0.007519], [-0.001978, 0.00755, -0.013793, -0.007125, 0.013573, -0.008007, 0.004317, 0.012686, -0.011441, 0.018617], [-0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0], [-0.014114, -0.005291, 0.027993, -0.01024, 0.004808, -0.006875, 0.032892, 0.008333, -0.08822, -0.025573], [-0.013064, -0.014307, 0.000823, -0.021282, -0.011179, -0.000154, 0.014447, -0.022662, -0.039274, 0.014359], [-0.005927, -0.005651, 0.001643, -0.009835, -0.009794, 0.002114, 0.010728, -0.015477, -0.029819, 0.007958], [0.10571, 0.086966, 0.107759, 0.276238, 0.154007, -0.04601, -0.082344, 0.458484, 0.183837, 0.106131], [-0.001654, -0.001311, 0.000413, -0.003349, 0.001407, 0.000363, 0.00254, -0.008159, -0.005517, -0.003567], [-0.001606, -0.002515, -0.001014, -0.006074, -0.006919, -0.002863, 0.000297, -0.025399, -0.002744, -0.007519], [-0.013902, -0.010528, 0.020596, -0.002248, 0.006695, 0.002549, 0.010443, 0.031627, -0.068464, -0.034603], [-0.001831, -0.001389, 0.002995, -0.002951, -0.002131, -0.00257, 0.003315, -0.009683, -0.009776, -0.006513], [0.000131, -0.000701, 0.003611, -0.000513, 0.001526, -0.001573, 0.002979, 0.002588, -0.008753, -0.000646], [-0.003308, -0.002621, 0.000825, -0.006699, 0.002814, 0.000726, 0.005081, -0.016318, -0.011035, -0.007134], [-0.011982, -0.006213, -0.002346, -0.0422, -0.023245, 0.031799, -0.008639, -0.010564, -0.039414, -0.015978], [-0.032762, -0.052001, 0.012552, -0.004351, 0.011888, 0.003532, 0.063311, 0.002953, 0.046485, 0.025888], [-0.062602, -0.093654, 0.057401, -0.017075, 0.025763, 0.106034, -0.059964, 0.11909, -0.074299, -0.052579], [-0.005653, -0.004255, -0.001516, -0.016564, -0.005282, 0.005715, 0.003208, -0.028583, -0.018752, -0.011088], [0.313811, -0.29838, 0.463304, -0.121827, -0.223701, -0.133323, -0.129145, 0.142787, -0.133639, 0.401945], [-0.095698, -0.11327, -0.009953, 0.015988, 0.028984, -0.007954, -0.259406, 0.166095, -0.046337, -0.126025], [-0.069831, -0.01384, 0.113382, 0.009622, 0.012808, 0.117098, 0.000541, 0.101663, -0.028644, -0.049342], [-0.031226, -0.036039, 0.004255, -0.012332, 0.006724, -0.038216, 0.013865, 0.018003, -0.025764, -0.021192], [-0.060759, -0.044436, -0.006119, -0.029597, 0.004825, -0.012406, -0.024791, -0.038655, -0.037864, 0.028517], [-0.003308, -0.002621, 0.000825, -0.006699, 0.002814, 0.000726, 0.005081, -0.016318, -0.011035, -0.007134], [-0.006294, -0.002912, -0.002593, -0.020246, -0.006492, 0.014589, 0.000681, -0.017274, -0.028521, -0.005615], [-0.01294, -0.004286, -0.011387, -0.049361, -0.02157, 0.039456, -0.004441, -0.011695, -0.044566, -0.016604], [-0.031103, -0.02147, 0.006046, -0.072297, -0.006906, 0.078024, -0.020848, -0.007564, -0.079337, -0.032683], [-0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0], [-0.065587, 0.030137, -0.079757, -0.080554, 0.038451, 0.097468, 0.032318, 0.041856, -0.106861, -0.06776], [0.279561, 0.192573, 0.0796, -0.070036, -0.086229, 0.220017, 0.68344, -0.062141, 0.133875, 0.160629], [-0.036517, 0.001411, -0.006179, -0.083443, -0.034537, 0.048011, -0.016232, -0.00416, -0.069666, -0.036554], [-0.009029, -0.005372, -0.000345, -0.02013, 0.013908, 0.002225, 0.017918, -0.036768, -0.018499, -0.034447], [-0.039159, 0.005051, -0.057309, -0.063465, 0.036357, 0.102737, 0.062534, -0.055048, -0.132096, -0.017046], [-0.001616, 0.002339, -0.001885, -0.000622, -0.000357, -0.00895, -0.003993, 0.000963, -0.000674, 0.005312], [-0.00141, -0.001593, -0.000511, -0.005855, -0.006984, -0.000819, -0.000416, -0.02905, -0.012332, -0.003703], [-0.004852, -0.007489, -0.003504, -0.001083, 0.002149, 0.005299, -0.015258, 0.005813, -0.007744, -0.006568], [-0.172531, -0.027211, -0.160006, 0.453982, -0.21116, -0.02632, 0.140498, 0.073055, -0.043304, 0.258928], [-0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0], [-0.148849, 0.181676, -0.237846, 0.41914, 0.031478, 0.056015, -0.018961, -0.106172, 0.185238, -0.048998], [-0.077486, -0.020766, -0.058647, 0.090722, 0.03574, -0.197558, -0.242582, -0.004327, 0.076907, 0.120551], [0.353899, -0.061906, -0.248042, 0.213676, -0.235747, -0.198726, -0.099473, -0.385311, -0.120753, 0.227501], [-0.002036, 0.001866, -0.004599, -0.004752, 0.000983, -0.010946, -0.002189, 0.000593, 0.007009, 0.005224], [-0.014873, 0.010682, -0.002906, -0.004088, 0.014279, -0.027845, -0.043736, 0.002894, 0.012858, -0.006452], [-0.053322, 0.096525, -0.084344, -0.138646, -0.513891, 0.169337, 0.021859, 0.302113, 0.140981, -0.104432], [-0.045703, -0.046866, 4.3e-05, -0.077622, -0.044559, 0.109233, -0.127576, -0.048059, -0.016091, 0.00163], [-0.032676, -0.066832, -0.005082, -0.078708, 0.008245, 0.034385, -0.045222, -0.085553, 0.157906, 0.059661], [-0.007934, -0.008543, -0.006544, -0.022312, -0.019242, -0.001694, 0.003421, -0.062521, -0.018894, -0.01702], [-0.028145, -0.009445, -0.033167, -0.120419, -0.07285, 0.092261, -0.01869, -0.028405, -0.083229, -0.01137], [-0.009036, -0.008064, -0.004532, -0.023606, 0.008711, -0.003325, 0.027124, -0.0116, 0.031653, -0.001701], [-0.007154, -0.009802, -0.006445, -0.020408, -0.006559, 0.01279, -0.023899, -0.018508, 0.003616, 0.013341], [-0.020075, -0.006893, -0.018675, -0.02031, 0.011977, 0.018518, 0.03695, -0.02, 0.024422, -0.012968], [0.296192, 0.170927, -0.195723, 0.00915, 0.000431, 0.549288, -0.314223, 0.11748, 0.012884, 0.171554], [-0.010854, -0.008537, -0.005624, -0.027932, -0.011523, 0.017807, -0.000992, -0.013411, -0.033566, -0.010754], [-0.001542, -0.000376, -0.001835, -0.006828, -0.004637, 0.006421, -0.002022, 0.000435, -0.006346, -0.000194], [-0.012959, 0.014359, 0.032055, 0.003412, -0.020285, 0.028275, 0.009571, -0.026837, 0.018179, -0.035826], [-0.090012, -0.154049, -0.010238, 0.13423, 0.020179, -0.003139, 0.110445, 0.076834, 0.165244, 0.109618], [-0.017237, -0.023916, 0.024514, -0.01995, 0.016653, 0.031814, -0.014123, 0.035871, -0.048151, -0.034101], [-0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0], [-0.022085, -0.034114, 0.012344, -0.015009, 0.004869, 0.029304, 0.009592, -0.168393, -0.106702, -0.289329], [-0.048324, -0.058632, 0.015488, 0.003945, 0.044224, 0.030714, -0.03424, 0.034688, -0.093815, -0.104828], [-0.076658, -0.03717, 0.148988, 0.079704, 0.022114, 0.045379, 0.037354, 0.16443, 0.03047, -0.123931], [-0.172036, -0.148736, 0.234211, 0.140701, 0.034112, -0.218098, 0.287804, 0.102734, -0.008401, -0.091659], [0.235873, -0.207212, 0.228753, 0.33281, -0.242976, 0.083214, -0.070959, -0.073266, 0.256901, -0.40002], [-0.003938, 0.004916, 0.001069, -0.00526, -0.005809, -0.024792, -0.006752, -0.024801, 0.009014, -0.009491], [-0.004991, -0.003625, 0.004565, -0.005462, -0.004107, -0.008709, 0.002261, -0.018725, 0.001512, 0.005693], [-0.025233, -0.038774, -0.01322, -0.020358, -0.026706, 0.055969, -0.034764, 0.003999, -0.032912, 0.018705], [-0.021596, -0.011346, -0.008066, -0.029166, -0.029482, 0.034501, -0.021727, 0.009777, -0.045665, 0.015152], [-0.003169, -0.00343, -0.001164, -0.011473, -0.005252, 0.010872, -0.003715, -0.012416, -0.010751, -0.011511], [-0.007942, 0.001169, 0.000713, -0.011696, -0.007778, -0.001767, -0.002152, -0.003477, 0.004661, 0.015907], [-0.022054, 0.007476, -0.005084, 0.026991, -0.003231, 0.020921, -0.062316, -0.024248, -0.062915, -0.073874], [-0.028186, -0.020032, 0.005788, -0.01636, -0.017732, -0.002287, -0.03012, 0.022291, 0.018833, 0.004381], [-0.004399, 0.000186, 0.006258, -0.021578, -0.019496, 7.8e-05, -0.001932, -0.05441, 0.003238, -0.03871], [-0.010992, 0.008046, -0.012766, -0.022917, -0.016096, -0.014797, -0.021074, 0.001582, -0.027777, 0.027787]],
    ca: [2.436681, 4.685022, 7.715204, 11.516423, 14.446332, 17.552635, 20.851489, 23.194583, 25.578986, 28.345493, 33.111133, 36.154641, 40.694583, 45.467719, 51.5549, 57.164416],
    pa: [12.306509, 16.319908]
  };

  function clamp(x, lo, hi) {
    return x < lo ? lo : (x > hi ? hi : x);
  }

  /* Plomp (1978) power-sum effective audible SNR */
  function effectiveAudibleSnr(snrDb, snrQuietDb) {
    const ln10_10 = Math.LN10 / 10.0;
    const a = -snrDb * ln10_10;
    const b = -snrQuietDb * ln10_10;
    const m = Math.max(a, b);
    const lse = m + Math.log(Math.exp(a - m) + Math.exp(b - m));
    return -10.0 * lse / Math.LN10;
  }

  /* Build 10x10 precision matrix Lambda = inv(sdG^2 * 11^T + sdD^2 * I_10) via Sherman-Morrison */
  function buildCompoundSymmetryPrecision(sdG, sdD) {
    const vd = sdD * sdD;
    const vg = sdG * sdG;
    const diagAdd = 1.0 / vd;
    const offSub = vg / (vd * (vd + 10.0 * vg));
    const L = Array.from({ length: 10 }, () => new Array(10).fill(-offSub));
    for (let i = 0; i < 10; i++) L[i][i] += diagAdd;
    return L;
  }

  /* Exact 10x10 linear system solver & matrix inverse via Gauss-Jordan with partial pivoting */
  function invertAndSolve10(H, rhsOrNull) {
    const n = 10;
    const aug = Array.from({ length: n }, (_, r) => {
      const row = new Array(2 * n + (rhsOrNull ? 1 : 0)).fill(0);
      for (let c = 0; c < n; c++) row[c] = H[r][c];
      row[n + r] = 1.0;
      if (rhsOrNull) row[2 * n] = rhsOrNull[r];
      return row;
    });
    const width = aug[0].length;

    for (let col = 0; col < n; col++) {
      let piv = col;
      let maxAbs = Math.abs(aug[col][col]);
      for (let r = col + 1; r < n; r++) {
        const v = Math.abs(aug[r][col]);
        if (v > maxAbs) { maxAbs = v; piv = r; }
      }
      if (piv !== col) {
        const tmp = aug[col]; aug[col] = aug[piv]; aug[piv] = tmp;
      }
      const diag = Math.abs(aug[col][col]) < 1e-12 ? 1e-12 : aug[col][col];
      for (let j = col; j < width; j++) aug[col][j] /= diag;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = aug[r][col];
        if (f === 0) continue;
        for (let j = col; j < width; j++) aug[r][j] -= f * aug[col][j];
      }
    }

    const inv = Array.from({ length: n }, (_, r) => aug[r].slice(n, 2 * n));
    const sol = rhsOrNull ? aug.map(row => row[2 * n]) : null;
    return { inv, sol };
  }

  function shuffleInPlace(arr, randFn) {
    const rng = randFn || Math.random;
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
    return arr;
  }

  function parseConsonantFromFilename(filename) {
    const part = filename.split("_")[1] || "";
    return part.substring(1, part.length - 1).toUpperCase();
  }

  /*
   * Build Standard Fixed-SNR schedule (nBlocks * 20 trials = 100 trials):
   * Each 20-trial block contains all 20 unique testFilesBase stimuli exactly once,
   * partitioned into two 10-consonant half-blocks with opposite token parities and
   * boundary swaps so the same consonant NEVER plays twice in a row.
   */
  function buildStandardFixedSchedule(testFilesBase, nBlocks, randFn) {
    const rng = randFn || Math.random;
    const blocks = nBlocks || 5;
    const byConsTok = {};
    for (const f of testFilesBase) {
      const c = parseConsonantFromFilename(f);
      const tok = f.includes("_1_") ? 0 : 1;
      if (!byConsTok[c]) byConsTok[c] = [null, null];
      byConsTok[c][tok] = f;
    }

    const out = [];
    let prevCons = null;
    for (let b = 0; b < blocks; b++) {
      const parity = STIM_CONSONANTS.map(() => (rng() < 0.5 ? 0 : 1));
      for (let half = 0; half < 2; half++) {
        const order = shuffleInPlace([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], rng);
        if (prevCons !== null && STIM_CONSONANTS[order[0]] === prevCons) {
          const tmp = order[0]; order[0] = order[1]; order[1] = tmp;
        }
        for (const ci of order) {
          const c = STIM_CONSONANTS[ci];
          const tok = (parity[ci] + half) % 2;
          const fname = byConsTok[c][tok];
          const snrDb = CAL.refSnrDb[ci];
          out.push({
            consonant: c,
            consonantIdx: ci,
            tokenIdx: tok,
            stimulus: fname,
            clearFile: `VCV_a${c.toLowerCase()}a_${tok + 1}_60SNR.wav`,
            snrDb
          });
        }
        prevCons = STIM_CONSONANTS[order[9]];
      }
    }
    return out;
  }

  /*
   * Build Balanced Adaptive-SNR schedule (nTrials = 50 or 100):
   * Balanced 10-consonant randomized blocks, alternating Token 1 / Token 2 across
   * blocks, and zero consecutive consonant repeats across block boundaries.
   */
  function buildAdaptiveSchedule(nTrials, randFn) {
    const rng = randFn || Math.random;
    const nBlocks = Math.max(1, Math.ceil((nTrials || 50) / 10));
    const tokParity = Array.from({ length: 10 }, () => (rng() < 0.5 ? 0 : 1));
    const out = [];
    let prevCi = -1;

    for (let blk = 0; blk < nBlocks; blk++) {
      const perm = shuffleInPlace([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], rng);
      if (prevCi >= 0 && perm[0] === prevCi) {
        const tmp = perm[0]; perm[0] = perm[1]; perm[1] = tmp;
      }
      for (let pos = 0; pos < 10 && out.length < nTrials; pos++) {
        const ci = perm[pos];
        const tok = (tokParity[ci] + blk) % 2;
        const c = STIM_CONSONANTS[ci];
        out.push({
          consonant: c,
          consonantIdx: ci,
          tokenIdx: tok,
          clearFile: `VCV_a${c.toLowerCase()}a_${tok + 1}_60SNR.wav`
        });
      }
      prevCi = perm[9];
    }
    return out;
  }

  /*
   * Online Hierarchical Bayesian Adaptive-SNR Tracker (exact match to
   * adaptive_1db_hier in sim/exp_suprathreshold_efficiency.py).
   */
  function createAdaptiveTracker(options) {
    const opts = options || {};
    const snrBank = opts.snrBank || SNR_BANK_1DB;
    const mu0 = CAL.onlineMuTheta;
    const sdG = CAL.onlineSdG;
    const sdD = CAL.onlineSdDelta;
    const lam0 = buildCompoundSymmetryPrecision(sdG, sdD);
    const initSd = Math.sqrt(sdG * sdG + sdD * sdD);

    const thetaHat = new Array(10).fill(mu0);
    const thetaSd = new Array(10).fill(initSd);
    let postCov = Array.from({ length: 10 }, (_, r) =>
      Array.from({ length: 10 }, (__, c) => (r === c ? sdG * sdG + sdD * sdD : sdG * sdG))
    );
    const history = [];

    const GH_NODES = [-Math.sqrt(3.0), 0.0, Math.sqrt(3.0)];
    const GH_WEIGHTS = [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];

    function selectSnrForConsonant(ci) {
      const thCur = thetaHat[ci];
      const sdCur = thetaSd[ci];
      const tau0Cur = CAL.tau0[ci];
      let bestIdx = 0;
      let bestEfi = -1.0;

      for (let m = 0; m < snrBank.length; m++) {
        const s = snrBank[m];
        let efi = 0.0;
        for (let q = 0; q < 3; q++) {
          const thQ = thCur + GH_NODES[q] * sdCur;
          const eta = clamp(CAL.slope * (s - (tau0Cur + thQ)), -30.0, 30.0);
          const u = 1.0 / (1.0 + Math.exp(-eta));
          const p = clamp(CAL.gamma + CAL.span * u, 1e-6, 1.0 - 1e-6);
          const dp = CAL.slope * CAL.span * u * (1.0 - u);
          efi += GH_WEIGHTS[q] * (dp * dp) / (p * (1.0 - p));
        }
        if (efi > bestEfi) {
          bestEfi = efi;
          bestIdx = m;
        }
      }
      return snrBank[bestIdx];
    }

    function computeGlobalStats() {
      let meanRaw = 0.0;
      let covSum = 0.0;
      for (let i = 0; i < 10; i++) {
        meanRaw += thetaHat[i];
        for (let j = 0; j < 10; j++) covSum += postCov[i][j];
      }
      meanRaw /= 10.0;
      return {
        rawGlobalThetaDb: meanRaw,
        nhReferencedShiftDb: meanRaw - CAL.nhMeanShiftDb,
        globalSeDb: Math.sqrt(Math.max(covSum / 100.0, 0.01))
      };
    }

    function updateWithTrial(ci, tokenIdx, snrDb, isCorrect) {
      history.push({
        ci,
        tokenIdx,
        snrDb,
        corr: isCorrect ? 1.0 : 0.0
      });

      let wDiag = new Array(10).fill(0.0);
      for (let iter = 0; iter < 2; iter++) {
        const gLike = new Array(10).fill(0.0);
        wDiag = new Array(10).fill(0.0);

        for (let n = 0; n < history.length; n++) {
          const tr = history[n];
          const tSign = tr.tokenIdx === 0 ? 1.0 : -1.0;
          const base = CAL.slope * (tr.snrDb - CAL.tau0[tr.ci]) + tSign * CAL.tokHalfDiff[tr.ci];
          const eta = clamp(base - CAL.slope * thetaHat[tr.ci], -30.0, 30.0);
          const u = 1.0 / (1.0 + Math.exp(-eta));
          const p = clamp(CAL.gamma + CAL.span * u, 1e-7, 1.0 - 1e-7);
          const dpDth = -CAL.slope * CAL.span * u * (1.0 - u);
          const dLdp = tr.corr / p - (1.0 - tr.corr) / (1.0 - p);
          const d2Ldp2 = -tr.corr / (p * p) - (1.0 - tr.corr) / ((1.0 - p) * (1.0 - p));
          gLike[tr.ci] += dLdp * dpDth;
          wDiag[tr.ci] += Math.max(-d2Ldp2 * dpDth * dpDth, 0.0);
        }

        const grad = new Array(10).fill(0.0);
        const H = Array.from({ length: 10 }, (_, r) => lam0[r].slice());
        for (let r = 0; r < 10; r++) {
          let priorPull = 0.0;
          for (let c = 0; c < 10; c++) priorPull += (thetaHat[c] - mu0) * lam0[r][c];
          grad[r] = gLike[r] - priorPull;
          H[r][r] += wDiag[r];
        }
        const solved = invertAndSolve10(H, grad);
        for (let c = 0; c < 10; c++) {
          thetaHat[c] += clamp(solved.sol[c], -3.5, 3.5);
        }
        postCov = solved.inv;
      }

      /* Exact Sherman-Morrison per-consonant posterior SD (matches Python lines 356-360) */
      const invVd = 1.0 / (sdD * sdD);
      let sumWTilde = 0.0;
      for (let c = 0; c < 10; c++) {
        sumWTilde += wDiag[c] / (1.0 + sdD * sdD * wDiag[c]);
      }
      const varG = 1.0 / (1.0 / (sdG * sdG) + sumWTilde);
      for (let c = 0; c < 10; c++) {
        const denom = wDiag[c] + invVd;
        const ratio = invVd / denom;
        const varC = 1.0 / denom + ratio * ratio * varG;
        thetaSd[c] = Math.sqrt(Math.max(varC, 0.05));
      }

      return computeGlobalStats();
    }

    return {
      selectSnrForConsonant,
      updateWithTrial,
      getThetaHat: () => thetaHat.slice(),
      getThetaSd: () => thetaSd.slice(),
      getPostCov: () => postCov.map(r => r.slice()),
      getGlobalStats: computeGlobalStats
    };
  }

  /* Interpolate partial pure-tone audiogram onto the 10-frequency grid */
  function normalizeAudiogram10(rawAud) {
    if (!Array.isArray(rawAud)) return null;
    const known = [];
    for (let i = 0; i < 10; i++) {
      const v = rawAud[i];
      if (v !== null && v !== undefined && v !== "" && Number.isFinite(Number(v))) {
        known.push({ idx: i, logF: Math.log(AUD_FREQS_HZ[i]), hl: Number(v) });
      }
    }
    if (known.length < 2) return null;
    return AUD_FREQS_HZ.map((f, idx) => {
      const exact = known.find(k => k.idx === idx);
      if (exact) return exact.hl;
      const lf = Math.log(f);
      if (lf <= known[0].logF) return known[0].hl;
      if (lf >= known[known.length - 1].logF) return known[known.length - 1].hl;
      for (let k = 0; k < known.length - 1; k++) {
        if (lf >= known[k].logF && lf <= known[k + 1].logF) {
          const t = (lf - known[k].logF) / (known[k + 1].logF - known[k].logF);
          return known[k].hl + t * (known[k + 1].hl - known[k].hl);
        }
      }
      return known[known.length - 1].hl;
    });
  }

  /* Full 12-iteration Newton-Raphson Hierarchical MAP decoder (blind or Plomp audiogram-informed) */
  function runFullHierarchicalMap(trials, opts) {
    const usePlomp = Boolean(opts && opts.snrQuiet10);
    const mu0 = usePlomp ? CAL.supraPriorMuD : CAL.onlineMuTheta;
    const sdG = usePlomp ? CAL.supraPriorSdG : CAL.onlineSdG;
    const sdD = usePlomp ? CAL.supraPriorSdDelta : CAL.onlineSdDelta;
    const lam0 = buildCompoundSymmetryPrecision(sdG, sdD);
    const d = new Array(10).fill(mu0);
    let postCov = null;

    for (let it = 0; it < 14; it++) {
      const gLike = new Array(10).fill(0.0);
      const hDiag = new Array(10).fill(0.0);

      for (let n = 0; n < trials.length; n++) {
        const tr = trials[n];
        const ci = tr.consonantIdx;
        const tSign = tr.tokenIdx === 0 ? 1.0 : -1.0;
        const corr = tr.correct ? 1.0 : 0.0;
        const sEff = usePlomp ? effectiveAudibleSnr(tr.snrDb, opts.snrQuiet10[ci]) : tr.snrDb;
        const base = CAL.alpha0[ci] + CAL.slope * sEff + tSign * CAL.tokHalfDiff[ci];
        const eta = clamp(base - CAL.slope * d[ci], -32.0, 32.0);
        const u = 1.0 / (1.0 + Math.exp(-eta));
        const p = clamp(CAL.gamma + CAL.span * u, 1e-7, 1.0 - 1e-7);

        const dpDd = -CAL.slope * CAL.span * u * (1.0 - u);
        const d2pDd2 = CAL.slope * CAL.slope * CAL.span * u * (1.0 - u) * (1.0 - 2.0 * u);
        const dLdp = corr / p - (1.0 - corr) / (1.0 - p);
        const d2Ldp2 = -corr / (p * p) - (1.0 - corr) / ((1.0 - p) * (1.0 - p));

        gLike[ci] += dLdp * dpDd;
        hDiag[ci] += Math.max(-(d2Ldp2 * dpDd * dpDd + dLdp * d2pDd2), 0.0);
      }

      const grad = new Array(10).fill(0.0);
      const H = Array.from({ length: 10 }, (_, r) => lam0[r].slice());
      for (let r = 0; r < 10; r++) {
        let pull = 0.0;
        for (let c = 0; c < 10; c++) pull += (d[c] - mu0) * lam0[r][c];
        grad[r] = gLike[r] - pull;
        H[r][r] += hDiag[r];
      }
      const solved = invertAndSolve10(H, grad);
      postCov = solved.inv;
      let maxStep = 0.0;
      for (let c = 0; c < 10; c++) {
        const st = clamp(solved.sol[c], -3.0, 3.0);
        d[c] += st;
        if (Math.abs(st) > maxStep) maxStep = Math.abs(st);
      }
      if (maxStep < 1e-4) break;
    }

    let meanVal = 0.0;
    let covSum = 0.0;
    for (let i = 0; i < 10; i++) {
      meanVal += d[i];
      for (let j = 0; j < 10; j++) covSum += postCov[i][j];
    }
    meanVal /= 10.0;
    return {
      d10: d,
      meanRaw: meanVal,
      seGlobal: Math.sqrt(Math.max(covSum / 100.0, 0.01)),
      postCov
    };
  }

  /* Published PPTA prediction from 10x11 confusion count matrix */
  function predictPptaDbHL(counts10x11) {
    let total = 0.0;
    for (let r = 0; r < 10; r++) {
      for (let c = 0; c < 11; c++) total += counts10x11[r][c];
    }
    if (total <= 0) return null;
    const scale = 100.0 / total;
    /* Column-major (Fortran) flatten of 10x11 matrix into length-110 vector X */
    const X = new Array(110);
    let k = 0;
    for (let col = 0; col < 11; col++) {
      for (let row = 0; row < 10; row++) {
        X[k++] = counts10x11[row][col] * scale;
      }
    }
    /* XC = [1, X @ C] of length 11 */
    const XC = new Array(11).fill(0.0);
    XC[0] = 1.0;
    for (let j = 0; j < 10; j++) {
      let s = 0.0;
      for (let i = 0; i < 110; i++) s += X[i] * PPTA_MODEL.C[i][j];
      XC[j + 1] = s;
    }
    /* Z = XC @ B of length 15 */
    const nCut = PPTA_MODEL.B[0].length;
    let phat = [1.0];
    for (let m = 0; m < nCut; m++) {
      let z = 0.0;
      for (let j = 0; j < 11; j++) z += XC[j] * PPTA_MODEL.B[j][m];
      const pk = 1.0 / (1.0 + Math.exp(-clamp(z, -40.0, 40.0)));
      const next = [pk];
      for (let q = 0; q < phat.length; q++) next.push((1.0 - pk) * phat[q]);
      phat = next;
    }
    let rawA = 0.0;
    for (let q = 0; q < phat.length; q++) rawA += phat[q] * PPTA_MODEL.ca[q];
    return rawA + PPTA_MODEL.pa[1] - PPTA_MODEL.pa[0];
  }

  /* Miller-Nicely relative mutual information (%) for a phonetic feature */
  function computeFeatureTransmissionPct(counts10x11, featMap, nClasses) {
    /* Include "???" as a separate response bucket (nClasses) so abstentions count as lost info */
    const nRespBuckets = nClasses + 1;
    const joint = Array.from({ length: nClasses }, () => new Array(nRespBuckets).fill(0.0));
    let total = 0.0;

    for (let s = 0; s < 10; s++) {
      const fs = featMap[s];
      for (let r = 0; r < 11; r++) {
        const cnt = counts10x11[s][r];
        const fr = r < 10 ? featMap[r] : nClasses;
        joint[fs][fr] += cnt;
        total += cnt;
      }
    }
    if (total <= 0) return { infoPct: 0.0, accPct: 0.0 };

    const rowSum = joint.map(row => row.reduce((a, b) => a + b, 0.0));
    const colSum = new Array(nRespBuckets).fill(0.0);
    let hits = 0.0;
    for (let x = 0; x < nClasses; x++) {
      hits += joint[x][x];
      for (let y = 0; y < nRespBuckets; y++) colSum[y] += joint[x][y];
    }

    let hx = 0.0;
    for (let x = 0; x < nClasses; x++) {
      if (rowSum[x] > 0) {
        const px = rowSum[x] / total;
        hx -= px * Math.log2(px);
      }
    }
    if (hx <= 1e-12) return { infoPct: 0.0, accPct: (100.0 * hits) / total };

    let mi = 0.0;
    for (let x = 0; x < nClasses; x++) {
      for (let y = 0; y < nRespBuckets; y++) {
        const nxy = joint[x][y];
        if (nxy > 0) {
          const pxy = nxy / total;
          const px = rowSum[x] / total;
          const py = colSum[y] / total;
          mi += pxy * Math.log2(pxy / (px * py));
        }
      }
    }
    return {
      infoPct: clamp((100.0 * mi) / hx, 0.0, 100.0),
      accPct: (100.0 * hits) / total
    };
  }

  /* End-of-test unified summary for both Standard Fixed-SNR and Adaptive-SNR modes */
  function summarizeSession(params) {
    const trials = params.trials || [];
    const counts10x11 = params.counts10x11;
    const isAdaptive = Boolean(params.isAdaptive);
    const aud10 = normalizeAudiogram10(params.audiogramHL);

    const nTrials = trials.length;
    const nCorrect = trials.reduce((acc, t) => acc + (t.correct ? 1 : 0), 0);
    const pctCorrect = nTrials > 0 ? (100.0 * nCorrect) / nTrials : 0.0;
    const meanSnrDb = nTrials > 0 ? trials.reduce((acc, t) => acc + t.snrDb, 0.0) / nTrials : 0.0;

    const rts = trials.map(t => t.rtMs).filter(v => Number.isFinite(v) && v >= 0).sort((a, b) => a - b);
    const medianRtMs = rts.length > 0 ? rts[Math.floor(rts.length / 2)] : null;

    const blindFit = runFullHierarchicalMap(trials, null);
    const globalSnrLossDb = blindFit.meanRaw - CAL.nhMeanShiftDb;
    const globalSeDb = blindFit.seGlobal;

    const lfIdx = [0, 1, 2, 4, 8]; /* B, D, G, N, V */
    const hfIdx = [3, 5, 6, 7, 9]; /* K, S, SH, T, Z */
    const lfShiftDb = lfIdx.reduce((a, i) => a + blindFit.d10[i], 0.0) / 5.0 - CAL.nhMeanShiftDb;
    const hfShiftDb = hfIdx.reduce((a, i) => a + blindFit.d10[i], 0.0) / 5.0 - CAL.nhMeanShiftDb;

    let supraDeficitDb = null;
    let supraSeDb = null;
    let audibilityComponentDb = null;
    let pta124Db = null;

    if (aud10) {
      pta124Db = (aud10[3] + aud10[5] + aud10[7]) / 3.0;
      const snrQuiet10 = CUE_BAND_IDX.map((idxs, ci) => {
        const tBand = idxs.reduce((a, fi) => a + aud10[fi], 0.0) / idxs.length;
        return (CAL.ltass0SnrBandHL[ci] + CAL.ltassDeltaMDb) - CAL.betaAudiogram * tBand;
      });
      const plompFit = runFullHierarchicalMap(trials, { snrQuiet10 });
      supraDeficitDb = plompFit.meanRaw - CAL.nhMeanSupraDb;
      supraSeDb = plompFit.seGlobal;
      audibilityComponentDb = Math.max(0.0, globalSnrLossDb - supraDeficitDb);
    }

    const pptaDbHL = !isAdaptive && counts10x11 ? predictPptaDbHL(counts10x11) : null;
    const voicing = counts10x11 ? computeFeatureTransmissionPct(counts10x11, FEAT_VOICING, 2) : null;
    const manner  = counts10x11 ? computeFeatureTransmissionPct(counts10x11, FEAT_MANNER, 3) : null;
    const place   = counts10x11 ? computeFeatureTransmissionPct(counts10x11, FEAT_PLACE, 3) : null;

    return {
      isAdaptive,
      nTrials,
      nCorrect,
      pctCorrect,
      meanSnrDb,
      medianRtMs,
      globalSnrLossDb,
      globalSeDb,
      lfShiftDb,
      hfShiftDb,
      perConsonantShiftDb: blindFit.d10.map(v => v - CAL.nhMeanShiftDb),
      hasAudiogram: Boolean(aud10),
      pta124Db,
      audibilityComponentDb,
      supraDeficitDb,
      supraSeDb,
      pptaDbHL,
      features: voicing ? { voicing, manner, place } : null
    };
  }

  return {
    STIM_CONSONANTS,
    DISP_CONSONANTS,
    AUD_FREQS_HZ,
    SNR_BANK_1DB,
    CAL,
    effectiveAudibleSnr,
    parseConsonantFromFilename,
    buildStandardFixedSchedule,
    buildAdaptiveSchedule,
    createAdaptiveTracker,
    normalizeAudiogram10,
    runFullHierarchicalMap,
    predictPptaDbHL,
    computeFeatureTransmissionPct,
    summarizeSession
  };
});
