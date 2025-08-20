"""
Test suite for DataframeUtils class.

Run with: pytest test_utils.py -v
"""
import pytest
import pandas as pd
from unittest.mock import patch

# Assuming the utils module is importable
from google_sheets_helper.utils import DataframeUtils


@pytest.fixture
def utils():
    """Create a DataframeUtils instance for testing."""
    return DataframeUtils()


@pytest.fixture
def sample_mixed_data():
    """Create sample DataFrame with mixed data types."""
    return pd.DataFrame({
        'date_column': ['2024-01-15', '2024-02-20', '2024-03-25', None],
        'numeric_string': ['123', '456', '789', '101'],
        'float_string': ['12.5', '34.7', '56.9', '0.1'],
        'text_column': ['hello', 'world', 'test', 'data'],
        'mixed_column': ['123', 'abc', '456', 'def'],
        'timestamp_unix': [1642204800, 1645056000, 1647820800, 1650412800],  # Unix timestamps
        'already_numeric': [1, 2, 3, 4],
        'already_datetime': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'])
    })


class TestInitialization:
    """Test DataframeUtils initialization."""

    def test_default_initialization(self):
        """Test default parameter initialization."""
        utils = DataframeUtils()
        assert isinstance(utils, DataframeUtils), "Should be an instance of DataframeUtils"


class TestTextCleaning:
    """Test text cleaning functionality."""

    def test_clean_text_encoding_basic(self, utils):
        """Test basic text cleaning."""
        df = pd.DataFrame({
            'text_col': ['  hello  \n', 'world\r\n\t', '  test  '],
            'numeric_col': [1, 2, 3]
        })

        result = utils.clean_text_encoding(df)
        expected_text = ['hello', 'world', 'test']

        assert result['text_col'].tolist() == expected_text
        assert result['numeric_col'].tolist() == [1, 2, 3]  # Unchanged

    def test_clean_text_encoding_max_length(self, utils):
        """Test max length truncation."""
        df = pd.DataFrame({
            'text_col': ['a' * 300, 'short', 'b' * 100]
        })

        result = utils.clean_text_encoding(df, max_length=10)
        assert all(len(text) <= 10 for text in result['text_col'])

    def test_clean_text_encoding_no_normalization(self, utils):
        """Test with whitespace normalization disabled."""
        df = pd.DataFrame({
            'text_col': ['  hello  \n\n', 'world\r\n\t  ']
        })

        result = utils.clean_text_encoding(df, normalize_whitespace=False)
        # Should only strip, not normalize internal whitespace
        assert 'hello' in result['text_col'].iloc[0]
        assert 'world' in result['text_col'].iloc[1]


class TestMissingValues:
    """Test missing value handling."""

    def test_handle_missing_values_basic(self, utils):
        """Test basic missing value handling."""
        df = pd.DataFrame({
            'text_col': ['hello', None, '', 'world'],
            'numeric_col': [1.0, None, 3.0, 4.0]
        })

        result = utils.handle_missing_values(df, fill_object_values='MISSING')

        assert result['text_col'].tolist() == ['hello', 'MISSING', 'MISSING', 'world']
        assert pd.isna(result['numeric_col'].iloc[1])  # Numeric NaN preserved

    def test_handle_missing_values_with_numeric_fill(self, utils):
        """Test missing value handling with numeric fill."""
        df = pd.DataFrame({
            'numeric_col': [1.0, None, 3.0, 4.0]
        })

        result = utils.handle_missing_values(df, fill_numeric_values=0)
        assert result['numeric_col'].tolist() == [1.0, 0, 3.0, 4.0]


class TestColumnNameTransformation:
    """Test column name transformation."""

    def test_transform_column_names_snake_case(self, utils):
        """Test snake_case transformation."""
        df = pd.DataFrame({
            'campaign.name': [1, 2, 3],
            'ad_group.id': [4, 5, 6],
            'Campaign-Status': [7, 8, 9]
        })

        result = utils.transform_column_names(df, naming_convention='snake_case', remove_prefixes=True)
        expected_cols = ['name', 'id', 'campaign_status']

        assert result.columns.tolist() == expected_cols

    def test_transform_column_names_camel_case(self, utils):
        """Test camelCase transformation."""
        df = pd.DataFrame({
            'campaign.name': [1, 2, 3],
            'ad_group.status': [4, 5, 6]
        })

        result = utils.transform_column_names(df, naming_convention='camelCase', remove_prefixes=True)
        expected_cols = ['name', 'status']

        assert result.columns.tolist() == expected_cols

    def test_transform_column_names_no_prefix_removal(self, utils):
        """Test transformation without prefix removal."""
        df = pd.DataFrame({
            'campaign.name': [1, 2, 3],
            'ad.group.status': [4, 5, 6]
        })

        result = utils.transform_column_names(
            df,
            naming_convention='snake_case',
            remove_prefixes=False
        )
        expected_cols = ['campaign_name', 'ad_group_status']

        assert result.columns.tolist() == expected_cols

    def test_transform_column_names_invalid_convention(self, utils):
        """Test handling of invalid naming convention."""
        df = pd.DataFrame({'col1': [1, 2, 3]})

        with patch('logging.warning') as mock_warning:
            result = utils.transform_column_names(df, naming_convention='invalid')
            mock_warning.assert_called_once()
            # Should default to snake_case
            assert result.columns.tolist() == ['col1']


class TestDataSummary:
    """Test data summary functionality."""

    def test_get_data_summary(self, utils, sample_mixed_data):
        """Test comprehensive data summary."""
        summary = utils.get_data_summary(sample_mixed_data)

        # Check required keys
        required_keys = [
            'total_rows', 'total_columns', 'data_types', 'missing_values',
            'memory_usage_mb', 'numeric_columns', 'date_columns', 'text_columns'
        ]

        for key in required_keys:
            assert key in summary

        # Check basic values
        assert summary['total_rows'] == len(sample_mixed_data)
        assert summary['total_columns'] == len(sample_mixed_data.columns)
        assert isinstance(summary['memory_usage_mb'], (int, float))
        assert summary['memory_usage_mb'] > 0


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
