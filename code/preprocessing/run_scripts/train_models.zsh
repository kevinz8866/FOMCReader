dir='CNN'
echo 'Training emotion CNN for male'
python code/train_cnn.py -d CREMA-D RAVDESS -l emotion -c sad angry neutral happy disgust fearful -g Male -dir $dir
echo 'Training emotion CNN for female'
python code/train_cnn.py -d CREMA-D RAVDESS -l emotion -c sad angry neutral happy disgust fearful -g Female -dir $dir
echo 'Training gender CNN'
python code/train_cnn.py -d CREMA-D RAVDESS -l gender -c Male Female -g both -dir $dir