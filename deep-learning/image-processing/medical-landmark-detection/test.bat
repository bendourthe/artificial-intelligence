@ECHO OFF
call C:\Users\username\anaconda3\Scripts\activate.bat medical_landmark_detection_env
ECHO #-----------------------------------------------------------------------------#
ECHO # Python 3.10 Conda Environment Activated
ECHO #-----------------------------------------------------------------------------#
call cd C:\Users\username\medical-landmark-detection
call python main.py -d runs -r unet2d_runs -p test -m unet2d -l u2net -ds cervical_spine -c runs/unet2d_runs/cervical_spine/checkpoints/trained_model.pt
PAUSE nul | set /p "=<Hit Enter To Close Window>"