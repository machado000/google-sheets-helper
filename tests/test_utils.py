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
def custom_utils():
    """Create a DataframeUtils instance with custom parameters."""
    return DataframeUtils(
        date_detection_sample_size=50,
        numeric_detection_sample_size=50,
        date_success_threshold=0.6,
        numeric_success_threshold=0.7
    )


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
        assert utils.date_sample_size == 1000
        assert utils.numeric_sample_size == 1000
        assert utils.date_threshold == 0.7
        assert utils.numeric_threshold == 0.8

    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        utils = DataframeUtils(
            date_detection_sample_size=500,
            numeric_detection_sample_size=200,
            date_success_threshold=0.6,
            numeric_success_threshold=0.9
        )
        assert utils.date_sample_size == 500
        assert utils.numeric_sample_size == 200
        assert utils.date_threshold == 0.6
        assert utils.numeric_threshold == 0.9

    def test_repr(self):
        """Test string representation."""
        utils = DataframeUtils(date_detection_sample_size=100, numeric_detection_sample_size=50)
        expected = "<DataFrameUtils(date_sample=100, numeric_sample=50)>"
        assert repr(utils) == expected


class TestDateDetection:
    """Test date column detection functionality."""

    def test_identify_date_columns_basic(self, utils):
        """Test basic date column identification."""
        df = pd.DataFrame({
            'date_iso': ['2024-01-15', '2024-02-20', '2024-03-25'],
            'date_us': ['01/15/2024', '02/20/2024', '03/25/2024'],
            'not_date': ['hello', 'world', 'test'],
            'numeric': [1, 2, 3]
        })

        date_columns = utils._identify_date_columns(df, [])
        assert 'date_iso' in date_columns
        assert 'date_us' in date_columns
        assert 'not_date' not in date_columns
        assert 'numeric' not in date_columns

    def test_check_date_patterns(self, utils):
        """Test regex date pattern matching."""
        # Test various date formats
        test_cases = [
            (['2024-01-15', '2024-02-20', '2024-03-25'], True),  # ISO format
            (['01/15/2024', '02/20/2024', '03/25/2024'], True),  # US format
            (['15-Jan-2024', '20-Feb-2024', '25-Mar-2024'], True),  # DD-Mon-YYYY
            (['20240115', '20240220', '20240325'], True),  # YYYYMMDD
            (['hello', 'world', 'test'], False),  # Non-dates
            (['2024-01-15', 'hello', 'world'], False),  # Mixed (below threshold)
        ]

        for values, expected in test_cases:
            sample = pd.Series(values)
            result = utils._check_date_patterns(sample)
            assert result == expected, f"Failed for {values}"

    def test_check_pandas_datetime_success(self, utils):
        """Test pandas datetime parsing success."""
        # Successful parsing
        good_dates = pd.Series(['2024-01-15', '01/15/2024', 'Jan 15, 2024'])
        assert utils._check_pandas_datetime_success(good_dates) is True

        # Failed parsing
        bad_dates = pd.Series(['hello', 'world', 'test'])
        assert utils._check_pandas_datetime_success(bad_dates) is False

        # Mixed (below threshold)
        mixed_dates = pd.Series(['2024-01-15', 'hello', 'world', 'test'])
        assert utils._check_pandas_datetime_success(mixed_dates) is False

    def test_check_timestamp_patterns(self, utils):
        """Test Unix timestamp detection."""
        # Unix timestamps (seconds)
        unix_seconds = pd.Series([1642204800, 1645056000, 1647820800])
        assert utils._check_timestamp_patterns(unix_seconds) is True

        # Unix timestamps (milliseconds)
        unix_ms = pd.Series([1642204800000, 1645056000000, 1647820800000])
        assert utils._check_timestamp_patterns(unix_ms) is True

        # Non-timestamps
        non_timestamps = pd.Series([1, 2, 3])
        assert utils._check_timestamp_patterns(non_timestamps) is False

    def test_looks_like_date_column(self, utils):
        """Test overall date column detection logic."""
        # Clear date column
        date_series = pd.Series(['2024-01-15', '2024-02-20', '2024-03-25'])
        assert utils._looks_like_date_column(date_series) is True

        # Clear non-date column
        text_series = pd.Series(['hello', 'world', 'test'])
        assert utils._looks_like_date_column(text_series) is False

        # Empty series
        empty_series = pd.Series([], dtype='object')
        assert utils._looks_like_date_column(empty_series) is False


class TestNumericDetection:
    """Test numeric column detection functionality."""

    def test_looks_numeric_basic(self, utils):
        """Test basic numeric detection."""
        # Clearly numeric
        numeric_series = pd.Series(['123', '456', '789'])
        assert utils._looks_numeric(numeric_series) is True

        # Clearly non-numeric
        text_series = pd.Series(['hello', 'world', 'test'])
        assert utils._looks_numeric(text_series) is False

        # Mixed (below threshold)
        mixed_series = pd.Series(['123', 'hello', 'world', 'test'])
        assert utils._looks_numeric(mixed_series) is False

    def test_is_numeric_string(self, utils):
        """Test numeric string detection."""
        assert utils._is_numeric_string('123') is True
        assert utils._is_numeric_string('12.5') is True
        assert utils._is_numeric_string('$123.45') is True
        assert utils._is_numeric_string('1,234') is True
        assert utils._is_numeric_string('45%') is True
        assert utils._is_numeric_string('hello') is False
        assert utils._is_numeric_string('') is False
        assert utils._is_numeric_string(None) is False

    def test_should_be_integer(self, utils):
        """Test integer vs float determination."""
        # Should be integer
        int_series = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert utils._should_be_integer(int_series) is True

        # Should be float
        float_series = pd.Series([1.5, 2.7, 3.9])
        assert utils._should_be_integer(float_series) is False

        # Empty series
        empty_series = pd.Series([])
        assert utils._should_be_integer(empty_series) is False

        # Series with overflow values
        overflow_series = pd.Series([2**63, 2**63 + 1], dtype=object)  # Beyond int64 range
        assert utils._should_be_integer(overflow_series) is False


class TestFixDataTypes:
    """Test the main fix_data_types functionality."""

    def test_fix_data_types_basic(self, utils, sample_mixed_data):
        """Test basic data type fixing."""
        result = utils.fix_data_types(sample_mixed_data)

        # Check that date columns were converted
        assert pd.api.types.is_datetime64_any_dtype(result['date_column'])

        # Check that numeric string columns were converted
        assert pd.api.types.is_integer_dtype(result['numeric_string'])
        assert pd.api.types.is_float_dtype(result['float_string'])

        # Check that text columns remained as object
        assert result['text_column'].dtype == 'object'

        # Check that already-converted columns weren't changed
        assert pd.api.types.is_integer_dtype(result['already_numeric'])
        assert pd.api.types.is_datetime64_any_dtype(result['already_datetime'])

    def test_fix_data_types_with_skip_columns(self, utils, sample_mixed_data):
        """Test data type fixing with skip columns."""
        skip_cols = ['numeric_string', 'date_column']
        result = utils.fix_data_types(sample_mixed_data, skip_columns=skip_cols)

        # Skipped columns should remain as object
        assert result['numeric_string'].dtype == 'object'
        assert result['date_column'].dtype == 'object'

        # Non-skipped columns should be converted
        assert pd.api.types.is_float_dtype(result['float_string'])

    def test_fix_data_types_empty_dataframe(self, utils):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        result = utils.fix_data_types(empty_df)
        assert result.empty
        assert isinstance(result, pd.DataFrame)

    def test_fix_data_types_preserves_original(self, utils, sample_mixed_data):
        """Test that original DataFrame is not modified."""
        original_dtypes = sample_mixed_data.dtypes.copy()
        utils.fix_data_types(sample_mixed_data)

        # Original should be unchanged
        pd.testing.assert_series_equal(sample_mixed_data.dtypes, original_dtypes)


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


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_fix_data_types_with_errors(self, utils):
        """Test data type fixing with problematic data."""
        df = pd.DataFrame({
            'bad_dates': ['2024-01-15', '2024-02-20', '2024-03-25', 'not-a-date', None],
            'bad_numbers': ['123', '456', '789', '890', 'not-a-number']
        })

        with patch('logging.warning') as mock_warning:
            result = utils.fix_data_types(df)
            # Should log warnings but not crash
            assert mock_warning.call_count > 0
            assert isinstance(result, pd.DataFrame)

    def test_methods_with_empty_series(self, utils):
        """Test methods with empty pandas Series."""
        empty_series = pd.Series([])

        assert utils._looks_like_date_column(empty_series) is False
        assert utils._looks_numeric(empty_series) is False
        assert utils._check_date_patterns(empty_series) is False
        assert utils._check_pandas_datetime_success(empty_series) is False
        assert utils._check_timestamp_patterns(empty_series) is False

    def test_methods_with_all_null_series(self, utils):
        """Test methods with all-null pandas Series."""
        null_series = pd.Series([None, None, None], dtype='object')

        assert utils._looks_like_date_column(null_series) is False
        assert utils._looks_numeric(null_series) is False


class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_full_data_processing_pipeline(self, utils):
        """Test complete data processing workflow."""
        # Create complex test data
        df = pd.DataFrame({
            'messy_dates': ['  2024-01-15  \n', '2024/02/20', 'Mar 25, 2024'],
            'messy_numbers': ['  123  ', '$456.78', '1,234'],
            'messy_text': [' hello \n', 'world\t\t', '  test  '],
            'campaign.name': ['Campaign A', 'Campaign B', 'Campaign C'],
            'mixed_col': ['text1', 'text2', 'text3']
        })

        # Run full pipeline
        result = utils.fix_data_types(df)
        result = utils.clean_text_encoding(result)
        result = utils.handle_missing_values(result)
        result = utils.transform_column_names(result)

        # Verify results
        assert pd.api.types.is_datetime64_any_dtype(result['messy_dates'])
        assert pd.api.types.is_float_dtype(result['messy_numbers'])  # Due to decimals
        assert result['messy_text'].str.contains('\n').sum() == 0  # Cleaned
        assert 'name' in result.columns  # Column transformed

        # Get summary
        summary = utils.get_data_summary(result)
        assert summary['total_rows'] == 3
        assert summary['date_columns'] >= 1


# Parametrized tests for different data scenarios
class TestParametrized:
    """Parametrized tests for various data scenarios."""

    @pytest.mark.parametrize("date_format,expected", [
        (['2024-01-15', '2024-02-20'], True),  # ISO
        (['01/15/2024', '02/20/2024'], True),  # US
        (['15-Jan-2024', '20-Feb-2024'], True),  # DD-Mon-YYYY
        (['not', 'dates'], False),  # Non-dates
    ])
    def test_date_detection_formats(self, date_format, expected):
        """Test date detection with various formats."""
        utils = DataframeUtils()
        series = pd.Series(date_format)
        result = utils._check_date_patterns(series)
        assert result == expected

    @pytest.mark.parametrize("numeric_data,expected", [
        (['123', '456', '789'], True),  # Integers
        (['12.3', '45.6', '78.9'], True),  # Floats
        (['$123', '$456', '$789'], True),  # Currency
        (['hello', 'world'], False),  # Text
    ])
    def test_numeric_detection_formats(self, numeric_data, expected):
        """Test numeric detection with various formats."""
        utils = DataframeUtils()
        series = pd.Series(numeric_data)
        result = utils._looks_numeric(series)
        assert result == expected


# Performance tests
class TestPerformance:
    """Performance-related tests."""

    def test_large_dataframe_handling(self, utils):
        """Test handling of larger DataFrames."""
        # Create a larger DataFrame
        size = 10000
        df = pd.DataFrame({
            'dates': ['2024-01-15'] * size,
            'numbers': ['123'] * size,
            'text': ['hello'] * size
        })

        # Should complete without timeout or memory issues
        result = utils.fix_data_types(df)
        assert len(result) == size
        assert pd.api.types.is_datetime64_any_dtype(result['dates'])
        assert pd.api.types.is_integer_dtype(result['numbers'])


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
