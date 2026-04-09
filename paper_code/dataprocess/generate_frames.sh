for f in cam*.MP4; 
  do b=${f%.*}; 
  mkdir -p $b;
  ffmpeg -hwaccel cuda -i $b.MP4 -vsync 0 -q:v 2 $b/%05d.jpg
done