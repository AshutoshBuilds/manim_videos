
This project contains the code used to generate the explanatory math videos found on [3Blue1Brown](https://www.3blue1brown.com/).

This almost entirely consists of scenes generated using the library [Manim](https://github.com/3b1b/manim).  See also the community maintained version at [ManimCommunity](https://github.com/ManimCommunity/manim/).

Older projects may have code dependent on older versions of manim, and so may not run out of the box here.

Note, while the library Manim itself is [open source](https://opensource.org/osd) software and under the [MIT license](https://github.com/3b1b/manim/blob/master/LICENSE.md), the contents of this repository are available under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png

## Workflow

I made [this video](https://youtu.be/rbu7Zu5X1zI) to show more of how I use manim. Note that I'm using 3b1b/manim, not the community edition, some functionality may differ between the two. Aside from [installing manim itself](https://github.com/3b1b/manim?tab=readme-ov-file#installation), replicating the workflow involves some custom plugins with Sublime, the text editor I use.

If you use another text editor, the same functionality can be mimicked. The key is to make use of two facts.

- Running `manimgl (file name) (scene name) -se (line_number)` will drop you into an interactive mode at that line of the file, like a debugger, with an iPython terminal that can be used to interact with the scene.

- Within that interactive mode, if you enter "checkpoint_paste()" to the terminal, it will run whatever bit of code is copied to the clipboard. Moreover, if that copied code begins with a comment, the first time it sees that comment it will save the state of the scene at that point, and for all future calls on code beginning with the same comment, it will first revert to that state of the scene before running the code.
    - The argument "skip" of checkpoint_paste will mean it runs the code without animating, as if all run times set to 0.
    - The argument "record" of checkpoint_paste will cause whatever animations are run with that copied code to be rendered to file.

For my own workflow, I set up some keyboard shortcuts to kick off each of these commands. For those who want to try it out themselves, here's what's involved.

### Sublime-specific instructions

Install [Terminus](https://packagecontrol.io/packages/Terminus) (via package control). This is a terminal run within sublime, and it lets us write some plugins that take the state in sublime, like where your cursor is, what's highlighted, etc., and use that to run a desired command line instruction.

Take the files in the "sublime_custom_commands" sub-directory of this repo, and copy them into the Packages/User/ directory of your Sublime Application. This should be a directory with a path that looks something like /wherever/your/sublime/lives/Packages/User/

Add some keybindings to reference these commands. Here's what I have inside my key_bindings file, you can find your own under the menu Sublime Text -> Settings -> Keybindings

```
    { "keys": ["shift+super+r"], "command": "manim_run_scene" },
    { "keys": ["super+r"], "command": "manim_checkpoint_paste" },
    { "keys": ["super+alt+r"], "command": "manim_recorded_checkpoint_paste" },
    { "keys": ["super+ctrl+r"], "command": "manim_skipped_checkpoint_paste" },
    { "keys": ["super+e"], "command": "manim_exit" },
    { "keys": ["super+option+/"], "command": "comment_fold"},
```

For example, I bind the "command + shift + R" to a custom "manim_run_scene" command. If the cursor is inside a line of a scene, this will drop you into the interactive mode at that point of the scene. If the cursor is on the line defining the scene, it will copy to the clipboard the command needed to render that full scene to file.

I bind "command + R" to a "manim_checkpoint_paste" command, which will copy whatever bit of code is highlighted, and run "checkpoint_paste()" in the interactive terminal.

Of course, you could set these to whatever keyboard shortcuts you prefer.

# Installation & Setup

## Prerequisites

Before installing this project, ensure you have:

- **Python 3.7+** installed on your system
- **Git** for cloning the repository
- **FFmpeg** for video rendering (optional but recommended)
- **LaTeX** distribution for mathematical text rendering (optional - project works without it)

### Installing FFmpeg (Recommended)

**Windows (using Chocolatey):**
```bash
choco install ffmpeg
```

**macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg
```

### Installing LaTeX (Optional)

LaTeX is required for rendering mathematical equations and formulas. Without it, you can still create videos but mathematical text may not render properly.

**Windows:**
- Install [MikTeX](https://miktex.org/download) (recommended)
- Or use Chocolatey: `choco install miktex`

**macOS:**
```bash
brew install --cask mactex
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt install texlive-full

# CentOS/RHEL
sudo yum install texlive
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/3b1b/manim.git
cd manim
```

### 2. Create a Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the 3Blue1Brown Manim Library

```bash
pip install -e .
```

### 5. Configure the Project

The project uses a custom configuration file. Copy the example config:

```bash
cp custom_config.yml.example custom_config.yml
```

Edit `custom_config.yml` to set your preferred output directories and settings.

## Usage

### Basic Video Creation

1. **Create a Scene File**

Create a new Python file (e.g., `my_scene.py`) with your animation:

```python
from manim_imports_ext import *

class MyScene(Scene):
    def construct(self):
        # Your animation code here
        text = Text("Hello, Manim!")
        self.play(Write(text))
        self.wait(2)
```

2. **Render the Video**

**High Quality (for final videos):**
```bash
manimgl my_scene.py MyScene -w
```

**Low Quality (for testing):**
```bash
manimgl my_scene.py MyScene -l
```

**Preview Mode:**
```bash
manimgl my_scene.py MyScene -p
```

### Command Line Options

- `-w`: Render to high-quality video file
- `-l`: Render to low-quality video file (faster for testing)
- `-p`: Preview mode (opens window, doesn't save file)
- `-s`: Skip to the end and just save the final frame
- `-r [resolution]`: Set resolution (e.g., `-r 1080` for 1080p)
- `-f [fps]`: Set frame rate (default: 30)

### Example: Gradient Descent Video

We've included an example gradient descent explanation video. To render it:

```bash
manimgl gradient_descent_scene.py GradientDescentExplanation -w
```

This will create `videos/GradientDescentExplanation.mp4` in your output directory.

### Project Structure

```
manim_videos/
├── _2024/                    # Example scenes from 2024 videos
├── videos/                   # Output directory for rendered videos
├── custom_config.yml         # Configuration file
├── manim_imports_ext.py      # Custom imports and utilities
├── stage_scenes.py          # Scene staging utilities
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

### Configuration

Edit `custom_config.yml` to customize:

- **Output directories**: Where videos and images are saved
- **Resolution and quality**: Video resolution, frame rate
- **Colors and themes**: Default colors for your animations
- **Font settings**: Text rendering preferences

Example configuration:
```yaml
directories:
  base: "O:\\D temp\\manim_videos"
  output: "videos"

video:
  resolution: [1920, 1080]
  fps: 30
  background_color: "#1a1a2e"
```

### Troubleshooting

#### Common Issues

1. **"ModuleNotFoundError: No module named 'manimlib'"**
   - Make sure you're using the correct imports: `from manim_imports_ext import *`
   - Ensure you're in the virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux)

2. **"NameError: name 'Create' is not defined"**
   - Use `ShowCreation()` instead of `Create()` for this version of Manim

3. **LaTeX/Math Text Not Rendering**
   - Install a LaTeX distribution (see Prerequisites)
   - Or use `Text()` objects instead of `Tex()` for simple text

4. **Video Won't Render**
   - Check that FFmpeg is installed and in your PATH
   - Try low-quality render first: `manimgl scene.py SceneName -l`
   - Check the console output for specific error messages

5. **Slow Rendering**
   - Use low-quality mode for testing: `-l` flag
   - Reduce resolution in config: `resolution: [1280, 720]`
   - Use preview mode: `-p` flag

#### Getting Help

- Check the [Manim documentation](https://docs.manim.community/)
- Visit the [Manim Community Discord](https://discord.gg/manim)
- Browse existing scenes in the `_2024/` directory for examples
- Watch 3Blue1Brown's [workflow video](https://youtu.be/rbu7Zu5X1zI)

## Development Workflow

For advanced users interested in the development workflow used to create 3Blue1Brown videos:

Copyright © 2024 3Blue1Brown
