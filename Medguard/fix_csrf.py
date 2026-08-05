import os

# Read the file
with open('templates/base.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the CSRF token usage
old_str = "'X-CSRFToken': '{{ csrf_token }}'"
new_str = "'X-CSRFToken': getCSRFToken()"

content = content.replace(old_str, new_str)

# Write back
with open('templates/base.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("CSRF token fix applied successfully!")

# Now add the getCSRFToken function if it doesn't exist
if 'function getCSRFToken()' not in content:
    print("Adding getCSRFToken function...")
    # Find the script tag and add the function after toggleSidebar
    func_to_add = '''
        // Function to get CSRF token from cookie
        function getCSRFToken() {
            const name = 'csrftoken';
            let cookieValue = null;
            if (document.cookie && document.cookie !== '') {
                const cookies = document.cookie.split(';');
                for (let i = 0; i < cookies.length; i++) {
                    const cookie = cookies[i].trim();
                    if (cookie.substring(0, name.length + 1) === (name + '=')) {
                        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                        break;
                    }
                }
            }
            return cookieValue;
        }

'''
    # Insert after the first closing brace of toggleSidebar function
    old_script = '''        function toggleSidebar() {
            document.getElementById('sidebar').classList.toggle('open');
        }

        // Close sidebar'''
    
    new_script = '''        function toggleSidebar() {
            document.getElementById('sidebar').classList.toggle('open');
        }
''' + func_to_add + '''        // Close sidebar'''
    
    content = content.replace(old_script, new_script)
    
    with open('templates/base.html', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("getCSRFToken function added successfully!")

