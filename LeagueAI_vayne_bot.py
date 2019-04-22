from LeagueAI_helper import input_output, LeagueAIFramework, detection
import time
import cv2

####### TODO ########
#- Implement the detection stuff
#- Implement the vayne bot
#- Add the interaction stuff like clilcking and so on to the input_output class
#- Make it so that you can define a rectangle in which you want to record
#- check if the rgb flip trick actually corrects the error, i think the old implementation used the bgr image
#####################

####### Params ######
# Show the AI view or not:
show_window = True
# Output window size
output_size = int(3440/3), int(1440/3)
# To record the desktop use:
#IO = input_output(input_mode='desktop', SCREEN_WIDTH=1080, SCREEN_HEIGHT=1920)
# If you want to use the webcam as input use:
#IO = input_output(input_mode='webcam')
# If you want to use a videofile as input:
IO = input_output(input_mode='videofile', video_filename='videos/video.mp4')
####################

LeagueAI = LeagueAIFramework(config_file="cfg/LeagueAI.cfg", weights="weights/04_16_LeagueAI.weights", names_file="cfg/LeagueAI.names", classes_number = 5, resolution=960, threshold = 0.55, cuda=True, draw_boxes=True)

while True:
    start_time = time.time()
    # Get the current frame from either a video, a desktop region or webcam (for whatever reason)
    frame = IO.get_pixels()
    # Get the list of detected objects and their positions
    objects = LeagueAI.get_objects(frame)
    for o in objects:
        o.toString()
    
    # TODO implement the vayne bot

    # Write fps
    cycle_time = time.time()-start_time
    cv2.putText(frame, "FPS: {}".format(str(round(1/cycle_time,0))), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    # Show the AI view
    if show_window:
        frame = cv2.resize(frame, output_size)
        cv2.imshow('LeagueAI', frame)
        if (cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break



