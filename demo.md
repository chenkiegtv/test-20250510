好的，这有一个 Google Colab 笔记本的代码单元格，用于演示 Gemini 模型的 OCR 功能。

这个脚本将：

安装必要的 Python 库。
提示你输入你的 Google AI API 密钥。
使用一个示例图片 URL（你可以替换成你自己的图片）。
调用 Gemini 模型来提取图片中的文本。
打印提取出来的文本。
请将以下代码粘贴到 Google Colab 的一个代码单元格中运行：

# @title Gemini OCR 功能演示 (Google Colab)

# 1. 安装必要的库
!pip install -q google-generativeai Pillow requests

# 2. 导入库
import google.generativeai as genai
from PIL import Image
import requests
import io
import os
from google.colab import userdata # Colab 特有的，用于安全处理密钥

# 3. 配置 API 密钥
# 推荐使用 Colab 的 Secret Manager 来存储你的 API 密钥。
# (1) 在 Colab 左侧边栏点击 "🔑" (Secrets) 图标。
# (2) 点击 "+ ADD A NEW SECRET"。
# (3) Name: GOOGLE_API_KEY, Value: 粘贴你的API密钥。
# (4) 勾选 "Notebook access"。
# 如果你不使用 Secret Manager，可以取消下面一行的注释，并直接粘贴密钥（但不推荐）
# GOOGLE_API_KEY = "YOUR_API_KEY"

try:
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        raise ValueError("API Key not found in Colab Secrets. Please add it.")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("API Key 配置成功！")
except Exception as e:
    print(f"获取 API 密钥失败或配置失败: {e}")
    print("请确保你已经在 Colab 的 Secrets 中正确设置了名为 'GOOGLE_API_KEY' 的密钥，并授予了 Notebook 访问权限。")
    # 或者，你可以取消下面代码块的注释，手动输入API Key (不推荐，因为密钥会显示在输出中)
    # if 'google.colab' in str(get_ipython()):
    #     from google.colab import output
    #     output.enable_custom_widget_manager()
    #     from IPython.display import display, Markdown
    #     api_key_input = input("请输入你的 Google AI API Key: ")
    #     if api_key_input:
    #         GOOGLE_API_KEY = api_key_input
    #         genai.configure(api_key=GOOGLE_API_KEY)
    #         print("API Key 已手动配置。")
    #     else:
    #         print("未输入API Key，脚本可能无法运行。")
    # else:
    #     print("无法自动获取API密钥，请手动在代码中配置。")


# 4. 加载 Gemini 模型
# 使用你在之前脚本中提到的 Gemini 2.5 Pro 模型名称，或者根据可用性选择最新的Pro模型。
# 例如 "gemini-1.5-pro-latest" 是一个非常强大的多模态模型。
# 如果你有特定的 "Gemini 2.5 Pro" 预览模型名称并且有权限访问，请使用那个名称。
# 这里我们继续使用你之前例子中的模型名，如果遇到问题，可以尝试 'gemini-1.5-pro-latest'
model_name_to_use = "gemini-1.5-pro-latest" # 或者你指定的 "gemini-2.5-pro-preview-05-06"
try:
    model = genai.GenerativeModel(model_name=model_name_to_use)
    print(f"成功加载模型: {model_name_to_use}")
except Exception as e:
    print(f"加载模型失败: {e}")
    print("请检查模型名称是否正确，以及你的API密钥是否有权访问该模型。")
    model = None # 确保模型未定义时不会在后续步骤出错

if model:
    # 5. 准备图像
    # 你可以使用网络图片的URL，或者上传本地图片到Colab
    # 示例1: 使用网络图片 URL
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/39/ReceiptSwiss.jpg/800px-ReceiptSwiss.jpg" # 瑞士收据示例
    # image_url = "https://www.gstatic.com/webp/gallery/1.jpg" # 另一个通用图片示例
    # image_url = "https://i.stack.imgur.com/U2V1s.png" # 一个包含代码的图片示例

    try:
        image_bytes = requests.get(image_url).content
        image = Image.open(io.BytesIO(image_bytes))
        print(f"图像已从 URL 下载: {image_url}")

        # （可选）在Colab中显示图片
        from IPython.display import display
        print("正在尝试显示图片...")
        display(image)

        # 6. 定义提示词 (Prompt)
        # 你可以根据需要调整提示词，例如要求结构化输出
        # prompt = "请提取这张图片中的所有文字。"
        prompt_structured = "请详细提取这张图片中的所有文字，并尝试以合理的结构化格式（例如JSON或Markdown列表）输出关键信息。"
        prompt_simple_ocr = "Extract all visible text from this image." # 英文提示，有时效果更佳

        # 选择一个提示词
        current_prompt = prompt_structured

        # 7. 调用模型进行 OCR
        print("\n正在调用 Gemini API 进行 OCR 处理，请稍候...")
        try:
            # 对于多模态输入，我们将提示和图像作为列表传递
            response = model.generate_content([current_prompt, image])

            # 8. 打印结果
            print("\n--- OCR 提取结果 ---")
            # response.text 包含模型生成的文本
            # 为了在Colab中更好地显示Markdown，我们可以使用 display(Markdown(...))
            from IPython.display import Markdown
            display(Markdown(response.text))

        except Exception as e:
            print(f"\n调用 Gemini API 时发生错误: {e}")
            if "API_KEY_INVALID" in str(e) or "API_KEY_SERVICE_BLOCKED" in str(e) or "permission" in str(e).lower():
                 print("错误提示与API密钥相关。请检查您的API密钥是否有效、是否已启用Generative Language API，以及是否有权访问所选模型。")
            elif "billing" in str(e).lower():
                print("错误提示与计费相关。请确保您的Google Cloud项目已设置有效的计费账户。")
            elif "Model service is not available for an on-demand model" in str(e):
                print(f"模型 '{model_name_to_use}' 可能当前不可用或需要特定权限/区域。尝试使用 'gemini-1.5-flash-latest' 或 'gemini-1.5-pro-latest'。")
            elif "Deadline" in str(e) or "timeout" in str(e).lower():
                print("请求超时。可能是网络问题或图片过大/复杂。尝试使用更小的图片或稍后再试。")
            else:
                print("发生未知错误。")

    except requests.exceptions.RequestException as e:
        print(f"下载图片失败: {e}")
    except Exception as e:
        print(f"处理图片时发生错误: {e}")
else:
    print("由于模型加载失败，无法继续进行OCR演示。")

如何使用这段代码：

打开 Google Colab：访问 colab.research.google.com 并创建一个新的笔记本。
设置 API 密钥：
推荐方式 (Secrets)：
在 Colab 左侧的工具栏中找到并点击钥匙图标 ("🔑" Secrets)。
点击 "+ ADD A NEW SECRET"。
Name: GOOGLE_API_KEY
Value: 粘贴你从 Google AI Studio (Makersuite) 或 Google Cloud Console 获取的 API 密钥。
确保 "Notebook access" 开关已启用。
备选方式 (直接粘贴 - 不推荐)：如果你不想使用 Secrets，可以取消代码中 GOOGLE_API_KEY = "YOUR_API_KEY" 这一行的注释，并将 "YOUR_API_KEY" 替换成你的实际密钥。但请注意，这样做会将你的密钥暴露在笔记本代码中，不安全。
粘贴代码：将上面提供的整块代码复制并粘贴到 Colab 笔记本的一个代码单元格中。
运行单元格：点击单元格左侧的“播放”按钮，或者按 Shift + Enter 来运行代码。
查看输出：
代码首先会安装必要的库。
然后它会尝试配置 API 密钥。
接着下载并显示示例图片。
最后，它会调用 Gemini API，并将提取到的文本以 Markdown 格式显示在单元格下方。
可以尝试修改的地方：

image_url: 你可以将其更改为任何其他公开可访问的图片 URL，以测试不同图片的效果。
prompt_structured 或 prompt_simple_ocr: 你可以修改提示词，看它如何影响输出的格式和内容。例如，你可以要求它只提取特定部分，或者以 JSON 格式输出。
model_name_to_use: 如果 gemini-2.5-pro-preview-05-06 (或者你使用的其他2.5 Pro模型) 遇到问题 (例如权限或可用性问题)，你可以尝试将其更改为 gemini-1.5-pro-latest 或 gemini-1.5-flash-latest，这些是目前比较稳定且强大的多模态模型。
