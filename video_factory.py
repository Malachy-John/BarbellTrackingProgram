import contour_track_21 as vp
import os
import pprint
import pandas as pd

video_folder = './videos'

video_info_dict = {}

pp = pprint.PrettyPrinter(indent=4)


for filename in os.listdir(video_folder):
    if filename.endswith(('.mp4', '.avi', '.MOV')):  # Add any other video formats as needed
        print(filename)
        video_path = os.path.join(video_folder, filename)


        processor = vp.VideoProcessor(video_path)

        average_speeds = processor.run()

        
        parts = filename.split('_')
        if len(parts) == 7:  # Ensure the filename has enough parts
            # Extract the components based on your naming convention
            video_id = parts[1]
            session_id = parts[2]
            person_id = parts[3]
            set_id = parts[4]
            weight = parts[5]
            camera_id = parts[6].split('.')[0]  # Assuming the next part is the camera ID, remove file extension

            if person_id not in video_info_dict:
                video_info_dict[person_id] = []  # Initialize with an empty list to hold video info dictionaries

            # Append extracted info to lists
            video_info = {
                'VideoID' : video_id,
                'PersonID': person_id,
                'SessionNumber': session_id,
                'Weight' : weight,
                'SetNumber' : set_id,
                'CameraID': camera_id,
                'Filename': filename,
                'AverageSpeeds' : average_speeds
            }

            video_info_dict[person_id].append(video_info)

for person_id, videos in video_info_dict.items():
    print(f"Person ID: {person_id}")
    for video_info in videos:
        print(f"Filename: {video_info['Filename']}")

video_info_list = []
for person_id, videos in video_info_dict.items():
    for video_info in videos:
        # You could add the PersonID here if it's not already included in video_info
        video_info['PersonID'] = person_id  # Ensure PersonID is part oqf each row
        video_info_list.append(video_info)

# Convert the list of dictionaries into a DataFrame
video_df = pd.DataFrame(video_info_list)        


video_df.to_csv("hello.csv")
pp.pprint(video_info_dict)
