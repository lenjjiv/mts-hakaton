import gradio as gr

# Исходный текст, который будет выводиться
text_subtitles = "Пример текста, размер которого можно изменять."

def update_text_size(size):
    """Обновляет размер текста в поле вывода."""
    return f"<p style='font-size: {size}px'>{text_subtitles}</p>"

with gr.Blocks() as demo:
    gr.Markdown("## Настроиваемый размер текста")
    
    text_output = gr.HTML(update_text_size(16))
    
    slider = gr.Slider(minimum=10, maximum=50, value=16, label="Размер текста")
    slider.change(update_text_size, inputs=slider, outputs=text_output)
    
    demo.launch()
