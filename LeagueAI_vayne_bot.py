from LeagueAI import input_output, LeagueAIFramework
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
####################

# To record the desktop use:
IO = input_output(input_mode='desktop', SCREEN_WIDTH=1080, SCREEN_HEIGHT=1920)
# If you want to use the webcam as input use:
#IO = input_output(input_mode='webcam')
# If you want to use a videofile as input:
#IO = input_output(input_mode='videofile', video_filename='video.mp4')

LeagueAI = LeagueAIFramework()

while True:
    start_time = time.time()
    # Get the current frame from either a video, a desktop region or webcam (for whatever reason)
    frame = IO.get_pixels()
    # Get the dictionary of detected objects and their positions
    objects = LeagueAI.get_objects()
    print(objects)

    # TODO implement the vayne bot

    # Write fps
    cycle_time = time.time()-start_time
    cv2.putText(frame, "FPS: {}".format(str(round(1/cycle_time,0))), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    # Show the AI view
    if show_window:
        cv2.imshow('LeagueAI', frame)
        if (cv2.waitKey(25) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break



