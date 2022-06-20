
for the in {0..12..1}
do
    for phi in {0..12..1}
    do
        echo "doing step "$the*$phi" of "12*12
        python new_ss.py $the $phi
    done
done
