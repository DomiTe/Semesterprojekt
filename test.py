import matplotlib.pyplot as plt

#ohne normalisierung
#lr=0.001
lr1= [3489.1356440891163, 3503.1175014148275, 3554.6675572203726, 3712.909456435475, 3704.8874339010326, 3813.4002372599216, 3561.1447275849337, 4095.7683989045036, 3617.716190530273, 4040.5753908875167, 4842.899157901369, 4359.407757537272, 4110.768947840951, 3895.169119609839, 3640.222732932539, 3623.595005175726, 3514.158293624213, 3543.5189205530205, 3671.6352037517245, 3802.339072986855, 4048.9610380163012, 4332.669621016929, 4663.5282224904495, 5023.932458710252, 4582.45951756044, 4212.96330222371, 3925.2528567354625, 3704.5082165686235, 3556.245504676115]
#lr=0.0001
lr2= [3796.9498157684725, 4335.104681199597, 4655.2041499899415, 4996.4042898975795, 5555.724241575145, 4123.701126418144, 4473.0672337305905, 3876.3581328810305, 4026.028709459116, 4214.765738780308, 5124.209579666388, 4819.662589918913, 14821.61265784168, 14647.093345447793, 14477.039086817345, 14310.136834257393, 14144.646207488859, 13982.05399003431, 13820.90946044231, 13659.704900718909, 13497.686573730192, 13338.165777149212, 13183.958128846562, 13030.706767075199, 12880.666060114689, 12731.659194056841, 12581.490126328612, 12432.742201869341, 12285.798157463656]

#mit normalisierung
#lr=0.0001
lr3= [0.0008382965864061025, 0.0004120035433072342, 0.0008073634228105939, 0.0009675362985357157, 0.00042272090151278545, 0.0007034448641633017, 0.0007326479688727934, 0.0006484607456030085, 0.0008425504804852914, 0.0006655347709267031, 0.000607349823897596, 0.000994288926257899, 0.0004373288592961409, 0.0009459108003394348, 0.0006627910153810662, 0.000612708719858497, 0.0009915488786259102, 0.0004349998254285483, 0.0009452270209959518, 0.0006758370571952929, 0.0005895409811589599, 0.001003003301211998, 0.0004443735139954142, 0.0009137309128621485, 0.00070827910669022, 0.0005728700437223835, 0.001009024019315143, 0.00044856958283176284, 0.0009107086035669647]
#lr=0.000001
lr4 = [0.014194730297943242, 0.010082707731142834, 0.007695957855440056, 0.005953613145568898, 0.004766346330964758, 0.003223627631580979, 0.001896257496261827, 0.001129315096069861, 0.0006670687966885169, 0.00037300472366197975, 0.0005972768770153706, 0.0009897949763897323, 0.001615233142345292, 0.002023384777616117, 0.002054665914481589, 0.001927397129710671, 0.00192367033197544, 0.0021697720877739777, 0.0022998407813701956, 0.002645229823369771, 0.0032188503296195145, 0.003349530046194872, 0.003578289277428876, 0.004084190723618538, 0.0048891439282582536, 0.0055989839801228545, 0.006604027981676443, 0.005957855085747967, 0.005255184373905304]

plt.plot(lr4)
plt.show()
