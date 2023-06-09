data_dir="nyt_corpus/data/"
for year in $(ls $data_dir)
do
	mkdir -p Nytimes/$year
	for file in $data_dir$year/*.tgz
	do
		tar -xzvf $file -C Nytimes/$year
	done
done

data_dir="Nytimes/"
for year in $(ls $data_dir)
do
  echo moving files in $data_dir$year
	find $data_dir$year -name "*.xml" | xargs -i mv {} $data_dir$year
done