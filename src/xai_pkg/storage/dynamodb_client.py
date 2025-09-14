"""
Simple DynamoDB client for storing LLM explanations and evaluation results.
"""

import boto3
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from decimal import Decimal
from botocore.exceptions import ClientError, NoCredentialsError

from config.config import settings

logger = logging.getLogger(__name__)

def convert_floats_to_decimal(obj):
    """Recursively convert float values to Decimal for DynamoDB compatibility."""
    if isinstance(obj, float):
        # Handle special float values that DynamoDB doesn't support
        import math
        if math.isnan(obj):
            return "NaN"  # Convert NaN to string
        elif math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"  # Convert Infinity to string
        else:
            return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {key: convert_floats_to_decimal(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimal(item) for item in obj]
    else:
        return obj

class DynamoDBClient:
    def __init__(self, region_name: str = None):
        """Initialize DynamoDB client."""
        # Use region from config if not specified
        if region_name is None:
            region_name = settings.get("DYNAMODB", {}).get("region", "us-east-1")
        
        self.region_name = region_name
        
        try:
            self.dynamodb = boto3.resource('dynamodb', region_name=region_name)
            logger.info(f"🗄️  Connected to DynamoDB in {region_name}")
        except NoCredentialsError:
            logger.error("❌ AWS credentials not found. Please configure your AWS credentials.")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to connect to DynamoDB: {e}")
            raise
    
    def store(self, data: Dict[str, Any], table_name: str, overwrite: bool = True) -> bool:
        """Store a single item."""
        try:
            table = self.dynamodb.Table(table_name)
            # Convert floats to Decimals for DynamoDB compatibility
            item = convert_floats_to_decimal(dict(data))
            item['stored_at'] = datetime.now().isoformat()
            
            if overwrite:
                # Just overwrite
                table.put_item(Item=item)
            else:
                # Check if exists first
                key_name = self._get_key_name(table_name)
                if key_name and key_name in item:
                    existing = table.get_item(Key={key_name: item[key_name]})
                    if 'Item' in existing:
                        logger.debug(f"⏭️ Item exists, skipping: {item[key_name]}")
                        return True  # Consider it success
                
                # Store if doesn't exist
                table.put_item(Item=item)
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to store item: {e}")
            return False
    
    def _get_key_name(self, table_name: str) -> Optional[str]:
        """Get the primary key name for a table."""
        if "explanation" in table_name.lower():
            return "explanation_id"
        elif "evaluation" in table_name.lower():
            return "explanation_id"
        return None
    
    def clear_table(self, table_name: str) -> bool:
        """Clear all items from a table."""
        try:
            table = self.dynamodb.Table(table_name)
            
            # Get table schema to understand key structure
            table_description = table.meta.client.describe_table(TableName=table_name)
            key_schema = table_description['Table']['KeySchema']
            
            # Build key structure from schema
            hash_key = None
            range_key = None
            for key in key_schema:
                if key['KeyType'] == 'HASH':
                    hash_key = key['AttributeName']
                elif key['KeyType'] == 'RANGE':
                    range_key = key['AttributeName']
            
            if not hash_key:
                logger.error(f"Cannot determine hash key for table {table_name}")
                return False
            
            # Get all items
            response = table.scan()
            items = response.get('Items', [])
            
            # Continue scanning if there are more items
            while 'LastEvaluatedKey' in response:
                response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
                items.extend(response.get('Items', []))
            
            # Delete all items using correct key structure
            with table.batch_writer() as batch:
                for item in items:
                    if hash_key in item:
                        delete_key = {hash_key: item[hash_key]}
                        if range_key and range_key in item:
                            delete_key[range_key] = item[range_key]
                        batch.delete_item(Key=delete_key)
            
            logger.info(f"🗑️  Cleared {len(items)} items from table {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear table {table_name}: {e}")
            return False

    def store_batch(self, items: List[Dict[str, Any]], table_name: str, overwrite: bool = True, clear_first: bool = False) -> Dict[str, int]:
        """Store multiple items, grouping explanations by prediction_id."""
        if not items:
            return {'success': 0, 'failed': 0, 'skipped': 0}
        
        stats = {'success': 0, 'failed': 0, 'skipped': 0}
        
        # Check if this is the explanations table - if so, group by prediction_id
        if "explanation" in table_name.lower():
            # Group explanations by prediction_id
            grouped = {}
            for item in items:
                pred_id = item.get('prediction_id')
                if not pred_id:
                    stats['failed'] += 1
                    continue
                
                if pred_id not in grouped:
                    grouped[pred_id] = []
                grouped[pred_id].append(item)
            
            # Store each prediction_id group
            for pred_id, explanations in grouped.items():
                try:
                    # Convert to grouped format expected by evaluation script
                    grouped_item = {
                        'prediction_id': pred_id,
                        'explanations': explanations,
                        'stored_at': datetime.now().isoformat()
                    }
                    
                    if self.store(grouped_item, table_name, overwrite):
                        stats['success'] += len(explanations)
                    else:
                        stats['failed'] += len(explanations)
                        
                except Exception as e:
                    logger.error(f"❌ Failed to store grouped explanations for {pred_id}: {e}")
                    stats['failed'] += len(explanations)
        else:
            # For other tables (like evaluations), store individually
            for item in items:
                if self.store(item, table_name, overwrite):
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
        
        logger.info(f"📊 Batch: {stats['success']} stored, {stats['failed']} failed")
        return stats
    
    def get(self, key: Dict[str, Any], table_name: str) -> Optional[Dict[str, Any]]:
        """Get a single item by key."""
        try:
            table = self.dynamodb.Table(table_name)
            response = table.get_item(Key=key)
            return response.get('Item')
        except Exception as e:
            logger.error(f"❌ Failed to get item: {e}")
            return None
    
    def get_all_explanations(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all explanations from DynamoDB."""
        try:
            table = self.dynamodb.Table("llm_explanations")
            explanations = []
            
            # Handle limit=0 case (return empty list)
            if limit == 0:
                logger.info(f"📥 Retrieved 0 explanations from DynamoDB (limit=0)")
                return []
            
            # Scan the table
            scan_kwargs = {}
            if limit is not None:
                scan_kwargs['Limit'] = limit
                
            response = table.scan(**scan_kwargs)
            items = response.get('Items', [])
            explanations.extend(items)
            
            # Continue scanning if there are more items and no limit specified
            while 'LastEvaluatedKey' in response and limit is None:
                response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
                items = response.get('Items', [])
                explanations.extend(items)
            
            logger.info(f"📥 Retrieved {len(explanations)} explanations from DynamoDB")
            return explanations
            
        except Exception as e:
            logger.error(f"❌ Failed to get explanations from DynamoDB: {e}")
            return []

    def get_all_evaluation_results(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all evaluation results from DynamoDB."""
        try:
            table = self.dynamodb.Table("evaluation_results")
            results = []
            
            # Handle limit=0 case (return empty list)
            if limit == 0:
                logger.info(f"📥 Retrieved 0 evaluation results from DynamoDB (limit=0)")
                return []
            
            # Scan the table
            scan_kwargs = {}
            if limit is not None:
                scan_kwargs['Limit'] = limit
                
            response = table.scan(**scan_kwargs)
            items = response.get('Items', [])
            results.extend(items)
            
            # Continue scanning if there are more items and no limit specified
            while 'LastEvaluatedKey' in response and limit is None:
                response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
                items = response.get('Items', [])
                results.extend(items)
            
            logger.info(f"📥 Retrieved {len(results)} evaluation results from DynamoDB")
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get evaluation results from DynamoDB: {e}")
            return []
    
    def create_tables(self, explanation_table: str = "llm_explanations", 
                     evaluation_table: str = "evaluation_results"):
        """Create tables if they don't exist."""
        
        # Explanations table (key: explanation_id)
        self._create_table_if_missing(
            table_name=explanation_table,
            key_name="explanation_id"
        )
        
        # Evaluations table (key: explanation_id)  
        self._create_table_if_missing(
            table_name=evaluation_table,
            key_name="explanation_id"
        )
    
    def _create_table_if_missing(self, table_name: str, key_name: str):
        """Create a table if it doesn't exist."""
        try:
            table = self.dynamodb.Table(table_name)
            table.load()
            logger.info(f"✅ Table '{table_name}' exists")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.info(f"🏗️  Creating table '{table_name}'...")
                
                self.dynamodb.create_table(
                    TableName=table_name,
                    KeySchema=[{'AttributeName': key_name, 'KeyType': 'HASH'}],
                    AttributeDefinitions=[{'AttributeName': key_name, 'AttributeType': 'S'}],
                    BillingMode='PAY_PER_REQUEST'
                )
                
                # Wait for it
                table = self.dynamodb.Table(table_name)
                table.wait_until_exists()
                logger.info(f"✅ Created table '{table_name}'")
            else:
                raise