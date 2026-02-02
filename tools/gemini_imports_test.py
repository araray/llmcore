# standalone_gemini_import_test.py
"""
A standalone script to test the imports required for the
LLMCore Gemini provider, based on the import logic in
llmcore/providers/gemini_provider.py.

This helps diagnose if 'google-genai' and its key dependencies
like 'google-api-core' are correctly installed and accessible.
"""

import sys


def run_import_tests():
    """Runs a series of import tests and prints the results."""
    print("--- Starting Google Gemini API Import Test ---")
    print(f"Python version: {sys.version}\n")

    # Test 1: Core 'google.genai' and its direct submodules/classes
    google_genai_base_available = False
    google_genai_types_module_available = False
    genai_errors_module_available = False
    genai_api_error_class_available = False
    genai_client_class_available = False

    print("Attempting to import 'google.genai' and its core components...")
    try:
        import google.genai as genai

        print("  SUCCESS: 'google.genai' imported.")
        google_genai_base_available = True

        # Test genai.types
        try:
            from google.genai import types as genai_types

            # Try accessing a type to be sure
            _ = genai_types.ContentDict
            print("  SUCCESS: 'google.genai.types' (as genai_types) imported and accessible.")
            google_genai_types_module_available = True
        except ImportError:
            print(
                "  FAILURE: Could not import 'google.genai.types'. This is a critical part of 'google-genai'."
            )
        except AttributeError:
            print(
                "  FAILURE: 'google.genai.types' imported, but 'ContentDict' (or similar) not found. Module might be incomplete."
            )

        # Test genai.errors and APIError
        try:
            from google.genai import errors as genai_errors

            print("  SUCCESS: 'google.genai.errors' (as genai_errors) imported.")
            genai_errors_module_available = True
            try:
                from google.genai.errors import APIError as GenAIAPIError

                _ = GenAIAPIError("test")  # Instantiate to check
                print("  SUCCESS: 'google.genai.errors.APIError' imported and usable.")
                genai_api_error_class_available = True
            except ImportError:
                print("  FAILURE: Could not import 'APIError' from 'google.genai.errors'.")
            except Exception as e_api_error:
                print(f"  WARNING: 'APIError' imported but failed instantiation: {e_api_error}")

        except ImportError:
            print("  FAILURE: Could not import 'google.genai.errors'.")

        # Test genai.Client
        try:
            _ = genai.Client  # Access the class
            print("  SUCCESS: 'google.genai.Client' class is accessible.")
            genai_client_class_available = True
        except AttributeError:
            print("  FAILURE: 'google.genai.Client' class not found in 'google.genai' module.")

    except ImportError:
        print(
            "  CRITICAL FAILURE: Could not import the base 'google.genai' module. Is it installed?"
        )

    print("\nAttempting to import from 'google.api_core.exceptions'...")
    # Test 2: 'google.api_core.exceptions'
    google_api_core_exceptions_available = False
    google_api_error_core_class_available = False
    permission_denied_class_available = False
    invalid_argument_class_available = False
    try:
        from google.api_core.exceptions import GoogleAPIError as CoreGoogleAPIError
        from google.api_core.exceptions import InvalidArgument, PermissionDenied

        print(
            "  SUCCESS: 'google.api_core.exceptions' imported (CoreGoogleAPIError, PermissionDenied, InvalidArgument)."
        )
        google_api_core_exceptions_available = True

        # Test instantiation of these exceptions
        try:
            _ = CoreGoogleAPIError("test")
            print("  SUCCESS: 'CoreGoogleAPIError' is usable.")
            google_api_error_core_class_available = True
        except Exception as e_core_api:
            print(
                f"  WARNING: 'CoreGoogleAPIError' imported but failed instantiation: {e_core_api}"
            )

        try:
            _ = PermissionDenied("test")
            print("  SUCCESS: 'PermissionDenied' is usable.")
            permission_denied_class_available = True
        except Exception as e_pd:
            print(f"  WARNING: 'PermissionDenied' imported but failed instantiation: {e_pd}")

        try:
            _ = InvalidArgument("test")
            print("  SUCCESS: 'InvalidArgument' is usable.")
            invalid_argument_class_available = True
        except Exception as e_ia:
            print(f"  WARNING: 'InvalidArgument' imported but failed instantiation: {e_ia}")

    except ImportError:
        print(
            "  FAILURE: Could not import from 'google.api_core.exceptions'. This is a key dependency for error handling."
        )

    # Summary of critical components for GeminiProvider
    print("\n--- Summary for LLMCore GeminiProvider Requirements ---")
    all_critical_available = (
        google_genai_base_available
        and google_genai_types_module_available  # google.genai.types for ContentDict etc.
        and google_api_core_exceptions_available  # For PermissionDenied, InvalidArgument
    )

    print(
        f"  'google.genai' base module: {'AVAILABLE' if google_genai_base_available else 'MISSING/FAILED'}"
    )
    print(
        f"  'google.genai.types' module: {'AVAILABLE' if google_genai_types_module_available else 'MISSING/FAILED'}"
    )
    print(
        f"  'google.genai.errors' module: {'AVAILABLE' if genai_errors_module_available else 'MISSING/FAILED'}"
    )
    print(
        f"  'google.genai.errors.APIError' class: {'AVAILABLE' if genai_api_error_class_available else 'MISSING/FAILED'}"
    )
    print(
        f"  'google.genai.Client' class: {'AVAILABLE' if genai_client_class_available else 'MISSING/FAILED'}"
    )
    print(
        f"  'google.api_core.exceptions' module (for PermissionDenied, InvalidArgument): {'AVAILABLE' if google_api_core_exceptions_available else 'MISSING/FAILED'}"
    )

    if all_critical_available:
        print("\nCONCLUSION: All critical imports for GeminiProvider appear to be AVAILABLE.")
        print(
            "If GeminiProvider still fails, the issue might be elsewhere (e.g., runtime API key, network, specific model access)."
        )
    else:
        print(
            "\nCONCLUSION: One or more critical imports for GeminiProvider are MISSING or FAILED."
        )
        print(
            "Please check your 'google-genai' installation and its dependencies (like 'google-api-core', 'google-ai-generativelanguage')."
        )
        print(
            "Try reinstalling with: pip install --upgrade --force-reinstall google-genai google-api-core google-ai-generativelanguage"
        )

    print("\n--- Import Test Finished ---")


if __name__ == "__main__":
    run_import_tests()
