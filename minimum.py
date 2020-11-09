import numpy as np
from util.reportable import ReportPlot
import matplotlib.pyplot as plt
nlms_rmse=[ 77.01556715, 86.6902385,  90.67967495, 91.59999662, 91.17760126, 90.90711929,
  90.34028356 ,88.79719196, 86.07038805, 83.03951632, 79.58359395, 75.33483783,
  70.72060517 ,67.22167233, 64.27052913, 61.3239985 , 58.55319234, 56.53664135,
  54.95550357, 53.64220961, 52.54613158, 51.61593226, 50.77571985, 50.07342418,
  49.39548137, 48.60446543, 47.8848462 , 47.48043856, 47.13675454, 46.68916208,
  46.21061989, 45.76886982, 45.3363233 , 44.90606409, 44.43528289, 44.09734151,
  43.85109556, 43.5598015 , 43.22637313, 42.92737248, 42.67597795, 42.49915457,
  42.36660452, 42.2294998  ,42.04685562, 41.88908342, 41.77327547, 41.63460905,
  41.5096202 , 41.41879386 ,41.38343499, 41.30562996, 41.23839107, 41.14545876,
  40.97985566, 40.71581522 ,40.39282309, 40.14652231, 39.91143821, 39.65262059,
  39.37597536, 39.1080437  ,38.85454017, 38.69254932, 38.51497211, 38.35285324,
  38.21289688, 38.08465261 ,37.93282968, 37.78853429, 37.64454691, 37.4548278,
  37.26205689, 37.11124489, 36.97883233, 36.82167615, 36.63130832, 36.5179802,
  36.42300567, 36.34340816, 36.22363623, 36.0498365 , 35.81362578, 35.6280271,
  35.48475631, 35.3843762 , 35.30365661, 35.19779482, 35.04167095, 34.85240568,
  34.72077737, 34.62763428, 34.47724425, 34.31379836, 34.24841236, 34.22487264,
  34.24700067, 34.30077395, 34.27663112 ,34.16875119, 34.08747079, 34.00698646,
  33.84717119, 33.67782465, 33.53260497, 33.38076965, 33.21474085, 33.05964325,
  32.93921038 ,32.84012383 ,32.75267031, 32.66624165, 32.49726309, 32.35288714,
  32.27247978, 32.20340746, 32.09715926, 31.99888027, 31.91976882, 31.86888552,
  31.82441888, 31.78562496, 31.76997579, 31.78152534, 31.83647537, 31.90714388,
  31.93153889, 31.89861483, 31.89991649, 31.90873815, 31.90260081, 31.87819897,
  31.819781  , 31.75372076, 31.67751458, 31.58868772 ,31.53053528, 31.47259929,
  31.41596751, 31.3421203 , 31.26585893, 31.21501578 ,31.2055583 , 31.18321606,
  31.13815299, 31.1065348 , 31.09247887, 31.08493548, 31.09106628, 31.06773038,
  31.05328928 ,30.99233884]


nlms_pr=[-31.31137692, -28.94464595, -28.04475435, -27.8427433,  -27.93513422,
  -27.99450612, -28.11956013 ,-28.46408531, -29.08783502, -29.80477143,
  -30.65490652, -31.75217264, -33.01624857, -34.03103657, -34.92889265,
  -35.86745587, -36.7921402 , -37.49304897, -38.06032541, -38.54404978,
  -38.95692478, -39.31413098, -39.64235951, -39.92090544, -40.19352727,
  -40.51639343, -40.81471782, -40.98434294, -41.12964096, -41.32046556,
  -41.52652302, -41.71864294, -41.90856486, -42.09929141, -42.31008374,
  -42.46278522, -42.57479558, -42.70811202, -42.86180946, -43.00065185,
  -43.1181423 , -43.20120431, -43.26370152, -43.32855217, -43.4152632,
  -43.49047293, -43.54586416, -43.61238592, -43.67253838, -43.71636854,
  -43.73346823, -43.7711249 , -43.80372403, -43.84886082, -43.9295314,
  -44.05882019 ,-44.2181132 , -44.34044349, -44.45789952, -44.58801385,
  -44.7280319  ,-44.86457645, -44.99462891, -45.07817356, -45.17016079,
  -45.25450907 ,-45.32761254, -45.39483073, -45.47470159, -45.55091047,
  -45.62724321 ,-45.72827129, -45.83145114, -45.91253894, -45.98400389,
  -46.06915773 ,-46.17279652, -46.2347395 , -46.28679386, -46.33051828,
  -46.396507   ,-46.49266657, -46.62411114, -46.72799283, -46.80854472,
  -46.86516659, -46.91080425, -46.97082761, -47.05969631, -47.16796839,
  -47.24359763, -47.29726708, -47.38426223, -47.47924616, -47.517331,
  -47.53101526, -47.51800923, -47.48654935, -47.50054604, -47.5634997,
  -47.6110441 , -47.65822856, -47.75234685, -47.8525698 , -47.93890798,
  -48.02958597, -48.12922859, -48.222763  , -48.29568497, -48.35587471,
  -48.40914035, -48.46192587, -48.56558847, -48.65457919, -48.70428945,
  -48.74708367, -48.81312587, -48.8744002 , -48.9238525  ,-48.95570735,
  -48.98357888, -49.00792525, -49.01772657, -49.01041211, -48.97581979,
  -48.93143473, -48.91611794, -48.93672571, -48.93588576, -48.93033731,
  -48.93416954 ,-48.94946157, -48.98614056, -49.02770121, -49.07575522,
  -49.13191742 ,-49.16877477, -49.20556397, -49.24159064, -49.2886634,
  -49.3373921 , -49.36994795, -49.37601606, -49.39034891, -49.41928025,
  -49.43960858, -49.44866033, -49.45352557, -49.44959588, -49.46463073,
  -49.47394868, -49.51326395]
print(np.min(nlms_rmse))
print(np.min(nlms_pr))
rp = ReportPlot(title="RMSE vs Filter Length", xlabel="Number of Taps", ylabel="RMSE", size=14, ticksize=7)
plt.figure(figsize  = rp.figsize)
rp.plotPy(np.linspace(2, 3*51, 152, dtype=np.uint16), nlms_rmse, 'NLMS')

rp = ReportPlot(title="Signal Power Reduction vs Taps", xlabel="Number of Taps", ylabel="Gain (dB)", size=14, ticksize=7)
plt.figure(figsize  = rp.figsize)
rp.plotPy(np.linspace(2, 3*51, 152, dtype=np.uint16), nlms_pr, 'NLMS')
plt.show()
