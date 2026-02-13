"""Tests unitarios para src/data/steam_transforms.py.

Cubre los casos de borde críticos de cada función pública del módulo,
priorizando las que retornan valores silenciosos (0.0 o '') ante inputs
inválidos en lugar de lanzar excepciones.

Ejecutar con:
    pytest tests/test_steam_transforms.py -v
"""

import pytest

from src.data.steam_transforms import (
    extract_list_names,
    extract_tags_top,
    parse_languages,
    parse_owners_midpoint,
)


# ===========================================================================
# parse_owners_midpoint
# ===========================================================================

class TestParseOwnersMidpoint:
    """Tests para parse_owners_midpoint — función de mayor riesgo de regresión
    silenciosa ya que retorna 0.0 ante cualquier input inválido."""

    def test_standard_range(self):
        assert parse_owners_midpoint('20000 - 50000') == 35000.0

    def test_range_with_spaces(self):
        assert parse_owners_midpoint(' 20000 - 50000 ') == 35000.0

    def test_zero_range(self):
        assert parse_owners_midpoint('0 - 0') == 0.0

    def test_range_with_commas(self):
        """Formato con separadores de miles: '1,000,000 - 2,000,000'."""
        assert parse_owners_midpoint('1,000,000 - 2,000,000') == 1_500_000.0

    def test_large_range(self):
        assert parse_owners_midpoint('50000000 - 100000000') == 75_000_000.0

    def test_asymmetric_range(self):
        assert parse_owners_midpoint('0 - 20000') == 10_000.0

    def test_empty_string(self):
        assert parse_owners_midpoint('') == 0.0

    def test_whitespace_only(self):
        assert parse_owners_midpoint('   ') == 0.0

    def test_none_input(self):
        assert parse_owners_midpoint(None) == 0.0

    def test_non_string_int(self):
        assert parse_owners_midpoint(50000) == 0.0

    def test_non_numeric_text(self):
        assert parse_owners_midpoint('unknown - many') == 0.0

    def test_single_number_no_dash(self):
        """Sin guion: no es un rango válido."""
        assert parse_owners_midpoint('50000') == 0.0

    def test_return_type_is_float(self):
        result = parse_owners_midpoint('20000 - 50000')
        assert isinstance(result, float)


# ===========================================================================
# extract_list_names
# ===========================================================================

class TestExtractListNames:
    """Tests para extract_list_names."""

    def test_list_of_strings(self):
        assert extract_list_names(['Action', 'Indie']) == 'Action|Indie'

    def test_list_of_dicts_with_description(self):
        value = [{'description': 'RPG'}, {'description': 'Strategy'}]
        assert extract_list_names(value) == 'RPG|Strategy'

    def test_list_of_dicts_with_name_fallback(self):
        value = [{'name': 'Horror'}, {'name': 'Thriller'}]
        assert extract_list_names(value) == 'Horror|Thriller'

    def test_single_element_list(self):
        assert extract_list_names(['Action']) == 'Action'

    def test_empty_list(self):
        assert extract_list_names([]) == ''

    def test_string_passthrough(self):
        """Si ya es un string, se devuelve tal cual."""
        assert extract_list_names('Action|Indie') == 'Action|Indie'

    def test_none_returns_empty(self):
        assert extract_list_names(None) == ''

    def test_nan_returns_empty(self):
        import numpy as np
        assert extract_list_names(float('nan')) == ''
        assert extract_list_names(np.nan) == ''

    def test_mixed_list(self):
        """Lista con strings y dicts mezclados."""
        value = ['Action', {'description': 'RPG'}]
        assert extract_list_names(value) == 'Action|RPG'

    def test_return_type_is_str(self):
        assert isinstance(extract_list_names(['Action']), str)
        assert isinstance(extract_list_names(None), str)


# ===========================================================================
# extract_tags_top
# ===========================================================================

class TestExtractTagsTop:
    """Tests para extract_tags_top."""

    def test_basic_top3(self):
        tags = {'RPG': 500, 'Action': 800, 'Indie': 300}
        result = extract_tags_top(tags, n=3)
        assert result == 'Action|RPG|Indie'

    def test_top_n_less_than_available(self):
        tags = {'RPG': 500, 'Action': 800, 'Indie': 300, 'Strategy': 200}
        result = extract_tags_top(tags, n=2)
        assert result == 'Action|RPG'

    def test_top_n_greater_than_available(self):
        """Pedir más tags de los disponibles retorna todos."""
        tags = {'RPG': 100, 'Action': 200}
        result = extract_tags_top(tags, n=10)
        assert result == 'Action|RPG'

    def test_single_tag(self):
        assert extract_tags_top({'Indie': 999}, n=5) == 'Indie'

    def test_empty_dict(self):
        assert extract_tags_top({}, n=5) == ''

    def test_none_input(self):
        assert extract_tags_top(None) == ''

    def test_non_dict_list(self):
        assert extract_tags_top(['Action', 'RPG']) == ''

    def test_default_n_is_5(self):
        tags = {f'tag{i}': 100 - i for i in range(10)}
        result = extract_tags_top(tags)
        assert len(result.split('|')) == 5

    def test_return_type_is_str(self):
        assert isinstance(extract_tags_top({'RPG': 1}), str)
        assert isinstance(extract_tags_top(None), str)

    def test_ties_are_stable(self):
        """Ties deben resolverse de forma determinista."""
        tags = {'A': 100, 'B': 100, 'C': 100}
        result = extract_tags_top(tags, n=3)
        assert len(result.split('|')) == 3


# ===========================================================================
# parse_languages
# ===========================================================================

class TestParseLanguages:
    """Tests para parse_languages."""

    def test_list_of_strings(self):
        result = parse_languages(['English', 'Spanish', 'French'])
        assert result == 'English, Spanish, French'

    def test_string_passthrough(self):
        assert parse_languages('English, Spanish') == 'English, Spanish'

    def test_none_returns_empty(self):
        assert parse_languages(None) == ''

    def test_nan_returns_empty(self):
        import numpy as np
        assert parse_languages(float('nan')) == ''
        assert parse_languages(np.nan) == ''

    def test_single_language_list(self):
        assert parse_languages(['English']) == 'English'

    def test_empty_list(self):
        assert parse_languages([]) == ''

    def test_list_with_mixed_types(self):
        """Lista con elementos no-string se convierte a str."""
        result = parse_languages([1, 'English', None])
        assert 'English' in result

    def test_return_type_is_str(self):
        assert isinstance(parse_languages(['English']), str)
        assert isinstance(parse_languages(None), str)
