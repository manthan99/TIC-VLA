def draw_text_with_wrapping(img, text, position, font, font_scale, color, thickness, max_width, line_spacing=1.5):
    """
    Draws text with word wrapping.
    position: (x, y) - top-left corner for the text block.
    """
    x, y = position
    words = text.split(' ')
    lines = []
    current_line = []
    
    # Text wrapping logic
    for word in words:
        current_line.append(word)
        line_str = ' '.join(current_line)
        (w, h), _ = cv2.getTextSize(line_str, font, font_scale, thickness)
        if w > max_width and len(current_line) > 1:
            current_line.pop()
            lines.append(' '.join(current_line))
            current_line = [word]
    lines.append(' '.join(current_line))
    
    # Draw logic
    for i, line in enumerate(lines):
        # Calculate vertical position
        dy = int(h * line_spacing * i)
        cv2.putText(img, line, (x, y + dy + h), font, font_scale, color, thickness, cv2.LINE_AA)
        
    # Return bottom y coordinate of the text block for further layout
    total_height = int(h * line_spacing * (len(lines) - 1)) + h
    return y + total_height
