# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 14:10:49 2021

@author: t-congmu
"""




from transformers import pipeline
from transformers import XLMProphetNetTokenizer, XLMProphetNetForConditionalGeneration
import dash
from dash import dcc, html
from dash.dependencies import Input, Output


# checkpoint = 'microsoft/xprophetnet-large-wiki100-cased'
# checkpoint = '/Users/congmu/Documents/Project/wd/base/checkpoint-300000'
checkpoint = '/Users/congmu/Documents/Project/wd/klue_mrc/ProphetNet-Ko_Base/checkpoint-182'
model = XLMProphetNetForConditionalGeneration.from_pretrained(checkpoint)
tokenizer = XLMProphetNetTokenizer.from_pretrained(checkpoint)

translator = pipeline(
    'translation_ko_to_en',
    'Helsinki-NLP/opus-mt-ko-en'
    # 'facebook/mbart-large-cc25'
)


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = 'Demo'
app.config.suppress_callback_exceptions = True
server = app.server


def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Question Generation"),
            html.H3("Demo for Question Generation (answer-agnostic)"),
            html.Div(
                id="intro",
                children="Please enter the context and choose parameters for generation. The model will generate several questions using different methods.",
            ),
        ],
    )

def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Context"),
            dcc.Textarea(id='context', style={'width': '100%', 'height': 300}),
            html.Br(),
            html.Br(),
            html.B("Number of Beams for Beam Search "),
            dcc.Input(id='num_beams', type='number', value=4, min=2, max=10, step=1),
            html.Br(),
            html.B("K for Top-K sampling "),
            dcc.Input(id='top_k', type='number', value=50, min=2, max=100, step=1),
            html.Br(),
            html.B("p for Top-p sampling "),
            dcc.Input(id='top_p', type='number', value=0.8, min=0, max=1, step=0.1),
        ],
    )


app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("microsoft_logo.png"))],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                # Patient Volume Heatmap
                html.Div(
                    id="patient_volume_card",
                    children=[
                        html.B("Input"),
                        html.Hr(),
                        html.Table([
                            html.Tr([html.Td('Context (Korean)'), html.Td('Translation (English)')]),
                            html.Tr([html.Td(id='context_ko'), html.Td(id='context_en')]),
                            ],
                            style={'width': '100%'}),
                    ],
                ),
                # Patient Wait time by Department
                html.Div(
                    id="wait_time_card",
                    children=[
                        html.B("Output"),
                        html.Hr(),
                        html.Table([
                            html.Tr([html.Td('Method'), html.Td('Question (Korean)'), html.Td('Translation (English)')]),
                            html.Tr([html.Td('Greedy Search'), html.Td(id='question_gs'), html.Td(id='translation_gs')]),
                            html.Tr([html.Td('Beam Search'), html.Td(id='question_bs'), html.Td(id='translation_bs')]),
                            html.Tr([html.Td('Top-K Sampling'), html.Td(id='question_topK'), html.Td(id='translation_topK')]),
                            html.Tr([html.Td('Top-p Sampling'), html.Td(id='question_topp'), html.Td(id='translation_topp')]),
                            ],
                            style={'width': '100%'}),
                            ],
                ),
            ],
        ),
    ],
)


@app.callback(
    Output(component_id='context_ko', component_property='children'),
    Output(component_id='context_en', component_property='children'),
    Output(component_id='question_gs', component_property='children'),
    Output(component_id='translation_gs', component_property='children'),
    Output(component_id='question_bs', component_property='children'),
    Output(component_id='translation_bs', component_property='children'),
    Output(component_id='question_topK', component_property='children'),
    Output(component_id='translation_topK', component_property='children'),
    Output(component_id='question_topp', component_property='children'),
    Output(component_id='translation_topp', component_property='children'),
    Input(component_id='context', component_property='value'),
    Input(component_id='num_beams', component_property='value'),
    Input(component_id='top_k', component_property='value'),
    Input(component_id='top_p', component_property='value')
)
def update_output_qg(context, num_beams, top_k, top_p):
    context_en = translator(context)[0]['translation_text']
    
    input_ids = tokenizer.encode(context, truncation=True, return_tensors="pt")
    
    output_ids_gs = model.generate(input_ids)
    output_ids_bs = model.generate(input_ids, num_beams=num_beams)
    output_ids_topK = model.generate(input_ids, do_sample=True, top_k=top_k)
    output_ids_topp = model.generate(input_ids, do_sample=True, top_p=top_p)

    question_gs = tokenizer.batch_decode(output_ids_gs, skip_special_tokens=True)[0]
    question_bs = tokenizer.batch_decode(output_ids_bs, skip_special_tokens=True)[0]
    question_topK = tokenizer.batch_decode(output_ids_topK, skip_special_tokens=True)[0]
    question_topp = tokenizer.batch_decode(output_ids_topp, skip_special_tokens=True)[0]

    translation_gs = translator(question_gs)[0]['translation_text']
    translation_bs = translator(question_bs)[0]['translation_text']
    translation_topK = translator(question_topK)[0]['translation_text']
    translation_topp = translator(question_topp)[0]['translation_text']
    
    return context, context_en, question_gs, translation_gs, question_bs, translation_bs, question_topK, translation_topK, question_topp, translation_topp


if __name__ == '__main__':
    app.run_server(debug=True)



