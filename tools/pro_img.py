import os
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def get_image_pairs(directory):
    images = os.listdir(directory)
    pairs = []
    for image in images:
        if image.endswith('.jpg') or image.endswith('.png'):
            base_name = os.path.splitext(image)[0]
            if base_name.endswith('_modi'):
                original_name = base_name[:-5]
                if original_name + '.jpg' in images or original_name + '.png' in images:
                    pairs.append((original_name, base_name))
    return pairs

def create_pdf(image_pairs, directory, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter

    for original, modi in image_pairs:
        original_path = os.path.join(directory, original + '.jpg')
        modi_path = os.path.join(directory, modi + '.jpg')

        if not os.path.exists(original_path):
            original_path = os.path.join(directory, original + '.png')
        if not os.path.exists(modi_path):
            modi_path = os.path.join(directory, modi + '.png')

        if os.path.exists(original_path) and os.path.exists(modi_path):
            original_img = Image.open(original_path)
            modi_img = Image.open(modi_path)

            original_img.thumbnail((width // 2, height // 2))
            modi_img.thumbnail((width // 2, height // 2))

            original_img_width, original_img_height = original_img.size
            modi_img_width, modi_img_height = modi_img.size

            c.drawImage(original_path, 0, height - original_img_height, original_img_width, original_img_height)
            c.drawImage(modi_path, width // 2, height - modi_img_height, modi_img_width, modi_img_height)

            c.showPage()

    c.save()


def get_image_triplets(directory):
    images = os.listdir(directory)
    triplets = []
    base_names = set()

    # Collect base names without extensions
    for image in images:
        if image.endswith('.jpg') or image.endswith('.png'):
            base_name = os.path.splitext(image)[0]
            base_names.add(base_name)

    # Find triplets
    for base_name in base_names:
        original_name = base_name
        modi_name = base_name + '_modi'
        pro_modi_name = base_name + '_pro_modi'

        if original_name in base_names and modi_name in base_names and pro_modi_name in base_names:
            triplets.append((original_name, modi_name, pro_modi_name))

    return triplets

def create_pdf_triplets(image_triplets, directory, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter

    for original, modi, pro_modi in image_triplets:
        original_path = os.path.join(directory, original + '.jpg')
        modi_path = os.path.join(directory, modi + '.jpg')
        pro_modi_path = os.path.join(directory, pro_modi + '.jpg')

        if not os.path.exists(original_path):
            original_path = os.path.join(directory, original + '.png')
        if not os.path.exists(modi_path):
            modi_path = os.path.join(directory, modi + '.png')
        if not os.path.exists(pro_modi_path):
            pro_modi_path = os.path.join(directory, pro_modi + '.png')

        if os.path.exists(original_path) and os.path.exists(modi_path) and os.path.exists(pro_modi_path):
            original_img = Image.open(original_path)
            modi_img = Image.open(modi_path)
            pro_modi_img = Image.open(pro_modi_path)

            original_img.thumbnail((width // 3, height // 3))
            modi_img.thumbnail((width // 3, height // 3))
            pro_modi_img.thumbnail((width // 3, height // 3))

            original_img_width, original_img_height = original_img.size
            modi_img_width, modi_img_height = modi_img.size
            pro_modi_img_width, pro_modi_img_height = pro_modi_img.size

            c.drawImage(original_path, 0, height - original_img_height, original_img_width, original_img_height)
            c.drawImage(modi_path, width // 3, height - modi_img_height, modi_img_width, modi_img_height)
            c.drawImage(pro_modi_path, 2 * (width // 3), height - pro_modi_img_height, pro_modi_img_width, pro_modi_img_height)

            c.showPage()

    c.save()
if __name__ == "__main__":
    directory = "/home/lianyaoxiu/lianyaoxiu/Infinity/outputs"
    output_pdf = "output.pdf"
    image_pairs = get_image_triplets(directory)
    print(image_pairs)
    create_pdf_triplets(image_pairs, directory, output_pdf)