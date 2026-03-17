The Paddy Dataset (UCI ID 1186) contains 2,790 instances across 46 features (45 predictors + target) from Tamil Nadu, India, capturing real-world paddy rice farming data with no missing values. It mixes categorical, integer, and continuous variables for yield prediction ML, ideal for your regression project on climate-impacted agriculture.
​
LINK TO DATASET: https://datalab-12.ics.uci.edu/dataset/1186/paddy+dataset
​

Farm & Input Features
Hectares (int64) measures cultivated land size. Agriblock, Variety (e.g., CO43, Ponmani), Soil Types (e.g., alluvial, clay), and Nursery (dry/wet) are categorical identifiers. Seedrate(in Kg) (int64) tracks seeding density; Nursery area (Cents) (int64) quantifies nursery land in local units. LP_Mainfield(in Tonnes) (float64) and LP_nurseryarea(in Tonnes) (int64) denote land preparation manure in main field/nursery. Timed inputs include DAP_20days (int64 kg), Weed28D_thiobencarb (int64 kg), Urea_40Days/Potassh_50Days (float64 kg), Micronutrients_70Days (int64 kg), Pest_60Day(in ml) (int64), and Trash(in bundles) (int64 post-harvest residue).
​

Weather Across Growth Stages
Features aggregate Days 1-30, 31-60, 61-90, 91-120:

Rainfall/Irrigation: 30DRain/30DAI (float64 mm); 30_50DRain/30_50DAI; 51_70Rain/51_70AI; 71_105Rain/71_105DAI.

Temperature: Min/Max temp per period (mostly float64).

Wind: Inst Wind Speed per period (int64 Knots); Wind Direction (str, e.g., N/S/E/W).

Humidity: Relative Humidity per period (mostly int64%).[user query context]

Target Variable
Paddy yield(in Kg) (int64) is the regression target

////////

These are all the features and their datatypes :

Hectares                                int64
Agriblock                                 str
Variety                                   str
Soil Types                                str
Seedrate(in Kg)                         int64
LP_Mainfield(in Tonnes)               float64
Nursery                                   str
Nursery area (Cents)                    int64
LP_nurseryarea(in Tonnes)               int64
DAP_20days                              int64
Weed28D_thiobencarb                     int64
Urea_40Days                           float64
Potassh_50Days                        float64
Micronutrients_70Days                   int64
Pest_60Day(in ml)                       int64
30DRain( in mm)                       float64
30DAI(in mm)                          float64
30_50DRain( in mm)                    float64
30_50DAI(in mm)                       float64
51_70DRain(in mm)                     float64
51_70AI(in mm)                        float64
71_105DRain(in mm)                    float64
71_105DAI(in mm)                      float64
Min temp_D1_D30                       float64
Max temp_D1_D30                         int64
Min temp_D31_D60                      float64
Max temp_D31_D60                        int64
Min temp_D61_D90                      float64
Max temp_D61_D90                      float64
Min temp_D91_D120                     float64
Max temp_D91_D120                     float64
Inst Wind Speed_D1_D30(in Knots)        int64
Inst Wind Speed_D31_D60(in Knots)       int64
Inst Wind Speed_D61_D90(in Knots)       int64
Inst Wind Speed_D91_D120(in Knots)      int64
Wind Direction_D1_D30                     str
Wind Direction_D31_D60                    str
Wind Direction_D61_D90                    str
Wind Direction_D91_D120                   str
Relative Humidity_D1_D30              float64
Relative Humidity_D31_D60               int64
Relative Humidity_D61_D90               int64
Relative Humidity_D91_D120              int64
Trash(in bundles)                       int64
Paddy yield(in Kg)                      int64
dtype: object

/// 
Categorical features:

Agriblock has 6 unique values.
Variety has 3 unique values.
Soil Types has 2 unique values.
Nursery has 2 unique values.
Wind Direction_D1_D30 has 6 unique values.
Wind Direction_D31_D60 has 5 unique values.
Wind Direction_D61_D90 has 5 unique values.
Wind Direction_D91_D120 has 6 unique values.