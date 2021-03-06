from flask import Flask, render_template, Response
from video_camera import VideoCamera


app = Flask(__name__)

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return(Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame'))




if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)