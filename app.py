# app.py
from flask import Flask, request, render_template, send_from_directory
import os
import uuid
from flask import jsonify  # Add this import
import matplotlib.pyplot as plt
from face2ppg import run_face2ppg

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PLOTS_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)



@app.route('/process', methods=['POST'])  # New endpoint
def process_video():
    try:
        video = request.files['video']
        if video.filename == '':
            return jsonify(error="No file selected"), 400
        
        # Save with webm extension
        video_filename = f"{uuid.uuid4().hex}.webm"
        video_path = os.path.join(UPLOAD_FOLDER, video_filename)
        video.save(video_path)

        # Your processing code
        times, ppg, heart_rate = run_face2ppg(video_path)
        
        # Generate plot
        plot_filename = f"{uuid.uuid4().hex}.png"
        plot_path = os.path.join(PLOTS_FOLDER, plot_filename)
        plt.figure()
        plt.plot(times, ppg)
        plt.xlabel("Time (s)")
        plt.ylabel("PPG Signal")
        plt.title("Extracted PPG Signal")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return jsonify({
            'heart_rate': round(heart_rate, 2),
            'plot_image': plot_filename
        })

    except Exception as e:
        return jsonify(error=f"Processing error: {str(e)}"), 500

@app.route('/results')
def show_results():
    heart_rate = request.args.get('hr')
    plot_image = request.args.get('plot')
    return render_template('result.html', 
                         heart_rate=heart_rate,
                         plot_image=plot_image)


@app.route('/', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        video = request.files['video']
        if video.filename == '':
            return "No file selected", 400
        # Corrected line: changed .mp4 to .webm
        video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.webm")
        video.save(video_path)

        try:
            times, ppg, heart_rate = run_face2ppg(video_path)
            os.remove(video_path)


            # Save plot image
            plot_filename = f"{uuid.uuid4().hex}.png"
            plot_path = os.path.join(PLOTS_FOLDER, plot_filename)
            plt.figure()
            plt.plot(times, ppg)
            plt.xlabel("Time (s)")
            plt.ylabel("PPG Signal")
            plt.title("Extracted PPG Signal")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

            return render_template('result.html', heart_rate=round(heart_rate, 2), plot_image=plot_filename)

        except Exception as e:
            return f"Processing error: {str(e)}", 500

    return render_template('upload.html')
        
        # Rest of your processing code remains the same
# @app.route('/', methods=['GET', 'POST'])
# def upload_video():
#     if request.method == 'POST':
#         video = request.files['video']
#         if video.filename == '':
#             return "No file selected", 400
#         video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.mp4")
#         video.save(video_path)

#         try:
#             times, ppg, heart_rate = run_face2ppg(video_path)

#             # Save plot image
#             plot_filename = f"{uuid.uuid4().hex}.png"
#             plot_path = os.path.join(PLOTS_FOLDER, plot_filename)
#             plt.figure()
#             plt.plot(times, ppg)
#             plt.xlabel("Time (s)")
#             plt.ylabel("PPG Signal")
#             plt.title("Extracted PPG Signal")
#             plt.tight_layout()
#             plt.savefig(plot_path)
#             plt.close()

#             return render_template('result.html', heart_rate=round(heart_rate, 2), plot_image=plot_filename)

#         except Exception as e:
#             return f"Processing error: {str(e)}", 500

#     return render_template('upload.html')



@app.route('/static/<filename>')
def serve_plot(filename):
    return send_from_directory(PLOTS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
