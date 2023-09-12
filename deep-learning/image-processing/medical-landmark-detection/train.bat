@ECHO OFF
call C:\Users\username\anaconda3\Scripts\activate.bat medical_landmark_detection_env
ECHO #-----------------------------------------------------------------------------#
ECHO # Python 3.10 Conda Environment Activated
ECHO #-----------------------------------------------------------------------------#
call cd C:\Users\username\medical-landmark-detection
call python main.py -d runs -r unet2d_runs -p train -m unet2d -e 50 -ds cervical_spine
PAUSE nul | set /p "=<Hit Enter To Close Window>"