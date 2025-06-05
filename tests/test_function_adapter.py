import unittest
import os
import json
from src.czbenchmarks.models.function_adapter import FunctionAdapter

def sample_function(input_path, output_path, params):
    with open(input_path, 'r') as infile:
        data = infile.read()

    processed_data = data.upper()  # Example processing

    with open(output_path, 'w') as outfile:
        outfile.write(processed_data)

    return "Processing complete"

class TestFunctionAdapter(unittest.TestCase):

    def setUp(self):
        self.input_file = "test_input.txt"
        self.output_file = "test_output.txt"
        self.params = {"key": "value"}

        with open(self.input_file, 'w') as f:
            f.write("test data")

        self.adapter = FunctionAdapter(file_path="test_function.py", function_name="sample_function")

        with open("test_function.py", 'w') as f:
            f.write("""
import os

def sample_function(input_path, output_path, params):
    with open(input_path, 'r') as infile:
        data = infile.read()

    processed_data = data.upper()

    with open(output_path, 'w') as outfile:
        outfile.write(processed_data)

    return "Processing complete"
""")

    def tearDown(self):
        os.remove(self.input_file)
        os.remove(self.output_file)
        os.remove("test_function.py")

    def test_run(self):
        result = self.adapter.run(self.input_file, self.output_file, self.params)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"], "Processing complete")

        with open(self.output_file, 'r') as f:
            output_data = f.read()

        self.assertEqual(output_data, "TEST DATA")

    def test_run_with_main_entrypoint(self):
        with open("test_function_main.py", 'w') as f:
            f.write("""
import os

def run(input_path, output_path, params):
    with open(input_path, 'r') as infile:
        data = infile.read()

    processed_data = data.lower()

    with open(output_path, 'w') as outfile:
        outfile.write(processed_data)

    return "Main processing complete"
""")

        adapter = FunctionAdapter(file_path="test_function_main.py")
        result = adapter.run(self.input_file, self.output_file, self.params)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"], "Main processing complete")

        with open(self.output_file, 'r') as f:
            output_data = f.read()

        self.assertEqual(output_data, "test data")

        os.remove("test_function_main.py")

    def test_run_with_main_entrypoint_no_function(self):
        with open("test_function_no_function.py", 'w') as f:
            f.write("""
import os

if __name__ == "__main__":
    def main(input_path, output_path, params):
        with open(input_path, 'r') as infile:
            data = infile.read()

        processed_data = data[::-1]  # Reverse the string

        with open(output_path, 'w') as outfile:
            outfile.write(processed_data)

        return "No explicit function processing complete"

    main("test_input.txt", "test_output.txt", {"key": "value"})
""")

        adapter = FunctionAdapter(file_path="test_function_no_function.py", function_name="__main__")
        result = adapter.run(self.input_file, self.output_file, self.params)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"], "Executed __main__ block")

        with open(self.output_file, 'r') as f:
            output_data = f.read()

        self.assertEqual(output_data, "atad tset")

        os.remove("test_function_no_function.py")

if __name__ == "__main__":
    unittest.main()
