import base64
import threading
import subprocess

from queue import Queue

import dash
import dash_bootstrap_components as dbc

from demo_helper import process

progress_queue = Queue(1)
progress_memory = 1
current_text = ""
current_g = dash.html.Div()


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout = dash.html.Div(children=[
    dash.dcc.Textarea(id='input-text',style={'width': '100%', 'height': 300}),
    dash.dcc.Interval(id='clock', interval=1000, n_intervals=0, max_intervals=-1),
    dbc.Progress(value=0, id="progress_bar"),
    dbc.Button("Start Work", id='start_work', n_clicks=0),
    dash.html.Div(id='text-output', style={'whiteSpace': 'pre'}),
    dash.html.Div(id="graph")
])


def draw_nx(g):
    svg = subprocess.run(
                ["dot", "./cache/cpg_reconstruction/demo.cpg", "-Tsvg"], 
                stdout=subprocess.PIPE,
                timeout=180).stdout.decode('utf-8')
    encoded_svg = base64.b64encode(svg.encode("utf-8")).decode()

    return dash.html.Img(src='data:image/svg+xml;base64,{}'.format(encoded_svg))


@app.callback(
    [
        dash.Output("progress_bar", "value"),
        dash.Output("text-output", "children"),
        dash.Output("graph", "children")
    ],
    [dash.Input("clock", "n_intervals")])
def progress_bar_update(n):
    global progress_memory
    global current_text
    global current_g
    while not progress_queue.empty():
        progress_memory, data = progress_queue.get()
        if progress_memory == 0:
            current_text = ""
            current_g = draw_nx(data)
        else:
            current_text = data
    progress_bar_val = progress_memory*100
    
    return (progress_bar_val, current_text, current_g)
   

@app.callback(
    [dash.Output("start_work", "n_clicks")],
    [dash.Input("start_work", "n_clicks"), dash.Input("input-text", "value")]
    )
def start_bar(n, text):
    if n==0 or progress_memory < 1 or len(text) < 2:
        return(0,)
    threading.Thread(target=start_work, args=(progress_queue,text)).start()
    return(0,)


def start_work(output_queue, text):
    process(text, output_queue)
    return(None)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True, use_reloader=True)