if [ "$#" -ne 1 ]
then
   echo "Usage: 'bash ./make_gif.sh <prefix>' [find ./prefix*.gf and ./prefix*.mesh to draw solutions using GLVis and save as gif file in prefix.gif]"
   exit 1
fi
if [[ "$1" == "-h" ]]
then
   echo "Usage: 'bash ./make_gif.sh <prefix>' [find ./prefix*.gf and ./prefix*.mesh to draw solutions using GLVis and save as gif file in prefix.gif]"
   exit 0
fi

prefix=$1

FILE="./tmp.glvs"
if test -f "$FILE"
then
    rm $FILE
fi

solutions=($prefix*.gf)
meshes=($prefix*.mesh)

nrSolutions=${#solutions[@]}
nrMeshes=${#meshes[@]}

echo "Found $nrSolutions solutions and $nrMeshes mesh files."
if [[ "$nrSolutions" == 0 ]]
then
    exit 0
fi
echo "solution ${meshes[0]} ${solutions[0]}" >> $FILE
echo "{" >> $FILE
echo "view 0 0" >> $FILE
echo "keys jlm**************" >> $FILE
echo "}" >> $FILE
echo "{" >> $FILE
for i in "${!solutions[@]}"
do
    if [[ "$nrMeshes" == "1" ]]
    then
        echo "    solution ${meshes[0]} ${solutions[$i]} screenshot ${solutions[$i]/.gf/.png}" >> $FILE
    else
        echo "    solution ${meshes[$i]} ${solutions[$i]} screenshot ${solutions[$i]/.gf/.png}" >> $FILE
    fi
done
echo "    keys q" >> $FILE
echo "}" >> $FILE

glvis -run $FILE
rm $FILE

convert -delay 10 -loop 0 $prefix-*.png $prefix.gif
echo "Results are saved in $prefix.gif."

exit 0