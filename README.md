# Goob Workshop
![](/assets/example.png)
This makes lightys. It can also make other people if you hit the load face button and load a face.

To use run start.bat and wait while it downloads a shitload of stuff. 

Many GB, may take a long time depending on your internet speed. 

Once it is finished itll load a gui. Images are saved in the outputs folder.

## Requirements

Python 3.10.9 (May work on other versions but was built on this)

Windows

Nvidia gpu with 10GB of video memory or higher. (Possibly could work on 8 but would be very close.)

This will work on linux if you manually make a conda env or venv and install the requirements from requirements.txt

Can probably be made to work on non-nvidia stuff if you sort out the torch backends and replace the .to("cuda") calls to to.("yourbackend)