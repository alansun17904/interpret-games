FILES=(
    "ann_5_params.csv"
    "ann_10_params.csv"
    "ann_15_params.csv"
    "ann_20_params.csv"
    "ann_25_params.csv"
    "ann_30_params.csv"
    "ann_40_params.csv"
    "ann_50_params.csv"
    "ann_60_params.csv"
    "ann_100_params.csv"
    "ann_200_params.csv"
)

STRATEGIES=(
    "Alternator"
    "Anti_Tit_For_Tat"
    "Bully"
    "Cooperator"
    "Cycler_DC"
    "Defector"
    "Suspicious_Tit_For_Tat"
    "Tit_For_Tat"
    "Win-Shift_Lose-Stay"
    "Win-Stay_Lose-Shift"
)

for file in ${FILES[@]}
do 
    python3 ann_recorder.py data/$file all
    for strategy in ${STRATEGIES[@]}
    do
        python3 ann_recorder.py data/$file "$strategy"
    done
done
