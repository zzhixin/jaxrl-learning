grep "UserWarning: key" warnings.log > compile.log
grep -A 1 "UserWarning: key" warnings.log > compile.log
sed -i -E 's/^.*UserWarning:\s//g' compile.log

grep "jit(train_one_step) with global shapes and types" warnings.log >> compile.log
sed -i -E '3,$s/\.\sArgument.*$//g' compile.log 
sed -i -E '3,$s/^.*with\sglobal\sshapes\sand\stypes\s//g' compile.log 
sed -i -E '3,$s/^\(//g' compile.log
sed -i -E '3,$s/\)$//g' compile.log

echo "The total number of all arguments' leaves are:" 
awk -F", " 'NR <= 1 { print NF }' compile.log
awk -F"ShapedArray" 'NR >= 2 { print NF-1 }' compile.log
