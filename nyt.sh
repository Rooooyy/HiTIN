root_dir="nyt/data/"
for year in $(ls $root_dir)
do
	mkdir -p nyt/NYT/$year
	for file in $root_dir$year/*.tgz
	do
		tar -xzvf $file -C nyt/NYT/$year
	done
done

root_dir="nyt/NYT/"
for year in $(ls $root_dir)
do
	find $root_dir$year -name "*.xml" | xargs -i mv {} $root_dir$year
done