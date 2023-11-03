NETWORKS=(
    5
    10
    15
    20
    25
    30
    40
    50
    60
    100
    200
)

STRATEGIES=(
    "all"
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

for network in ${NETWORKS[@]}
do
    for strategy in ${STRATEGIES[@]}
    do
       python3 generate_gpt_prompt.py data/prompt_header.txt data/"wa-$network-$strategy.pkl" data/"prompt-$network-$strategy.json" 
    done
done

