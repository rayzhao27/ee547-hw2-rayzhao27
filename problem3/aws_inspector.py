import argparse
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError, ProfileNotFound
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    class ClientError(Exception):
        def __init__(self, error_response):
            self.response = error_response
    class NoCredentialsError(Exception):
        pass
    class ProfileNotFound(Exception):
        pass


class AWSInspector:
    def __init__(self, region: str = None):
        self.region = region or self._get_default_region()
        self.session = None
        self.clients = {}
        
        if not self._validate_region(self.region):
            print(f"Invalid region: {self.region}")
            print("Please specify a valid AWS region (e.g., us-east-1, us-west-2, eu-west-1)")
            sys.exit(1)
        
    def _get_default_region(self) -> str:
        region = os.environ.get('AWS_DEFAULT_REGION')
        if region:
            return region
            
        try:
            session = boto3.Session()
            return session.region_name or 'us-east-1'
        except:
            return 'us-east-1'
    
    def _validate_region(self, region: str) -> bool:
        valid_regions = {
            'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2',
            'eu-west-1', 'eu-west-2', 'eu-west-3', 'eu-central-1', 'eu-north-1',
            'ap-southeast-1', 'ap-southeast-2', 'ap-northeast-1', 'ap-northeast-2',
            'ap-south-1', 'ca-central-1', 'sa-east-1',
            'af-south-1', 'ap-east-1', 'ap-northeast-3', 'ap-south-2', 'ap-southeast-3',
            'eu-south-1', 'me-south-1', 'us-gov-east-1', 'us-gov-west-1'
        }
        return region in valid_regions
    
    def authenticate(self) -> bool:
        if not BOTO3_AVAILABLE:
            print("boto3 is required but not installed!")
            print("Install with: pip install boto3")
            print("\nFor demonstration purposes, showing expected functionality...")
            return False
            
        try:
            self.session = boto3.Session()
            
            sts_client = self.session.client('sts')
            identity = sts_client.get_caller_identity()
            
            print(f"Authenticated as: {identity.get('Arn', 'Unknown')}")
            print(f"Using region: {self.region}")
            
            return True
            
        except NoCredentialsError:
            print("No AWS credentials found!")
            print("Please configure credentials using:")
            print("  1. aws configure")
            print("  2. Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)")
            sys.exit(1)
            
        except ProfileNotFound as e:
            print(f"AWS profile not found: {e}")
            sys.exit(1)
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'InvalidUserID.NotFound':
                print("Authentication failed: Invalid credentials")
            elif error_code == 'AccessDenied':
                print("Authentication failed: Access denied")
            else:
                print(f"Authentication failed: {e}")
            sys.exit(1)
            
        except Exception as e:
            print(f"Unexpected error during authentication: {e}")
            sys.exit(1)
    
    def _get_client(self, service: str):
        if service not in self.clients:
            self.clients[service] = self.session.client(service, region_name=self.region)
        return self.clients[service]
    
    def inspect_iam_users(self) -> List[Dict[str, Any]]:
        print("🔍 Inspecting IAM users...")
        
        try:
            iam = self._get_client('iam')
            users = []

            paginator = iam.get_paginator('list_users')
            for page in paginator.paginate():
                for user in page['Users']:
                    user_info = {
                        'username': user['UserName'],
                        'user_id': user['UserId'],
                        'arn': user['Arn'],
                        'created_date': user['CreateDate'].isoformat(),
                        'password_last_used': user.get('PasswordLastUsed', 'Never').isoformat() if user.get('PasswordLastUsed') != 'Never' else 'Never',
                        'attached_policies': [],
                        'inline_policies': []
                    }
                    
                    try:
                        attached_policies = iam.list_attached_user_policies(UserName=user['UserName'])
                        user_info['attached_policies'] = [
                            {
                                'policy_name': policy['PolicyName'],
                                'policy_arn': policy['PolicyArn']
                            }
                            for policy in attached_policies['AttachedPolicies']
                        ]
                    except ClientError as e:
                        if e.response['Error']['Code'] == 'AccessDenied':
                            print(f"  [WARNING] Access denied for IAM operations - skipping user enumeration")
                            return []
                        pass
                    
                    try:
                        inline_policies = iam.list_user_policies(UserName=user['UserName'])
                        user_info['inline_policies'] = inline_policies['PolicyNames']
                    except ClientError:
                        pass
                    
                    users.append(user_info)
            
            print(f"  Found {len(users)} IAM users")
            return users
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AccessDenied':
                print(f"  [WARNING] Access denied for IAM operations - skipping user enumeration")
                return []
            elif 'Timeout' in str(e) or 'timeout' in str(e).lower():
                print(f"  [WARNING] Network timeout for IAM operations - retrying once...")
                try:
                    return self.inspect_iam_users()
                except:
                    print(f"  [WARNING] Retry failed for IAM operations - skipping")
                    return []
            else:
                print(f"  [ERROR] Failed to access IAM operations: {e}")
                return []
    
    def inspect_ec2_instances(self) -> List[Dict[str, Any]]:
        print("🔍 Inspecting EC2 instances...")
        
        try:
            ec2 = self._get_client('ec2')
            instances = []
            
            paginator = ec2.get_paginator('describe_instances')
            for page in paginator.paginate():
                for reservation in page['Reservations']:
                    for instance in reservation['Instances']:
                        name = 'N/A'
                        for tag in instance.get('Tags', []):
                            if tag['Key'] == 'Name':
                                name = tag['Value']
                                break
                        
                        instance_info = {
                            'instance_id': instance['InstanceId'],
                            'name': name,
                            'instance_type': instance['InstanceType'],
                            'state': instance['State']['Name'],
                            'launch_time': instance.get('LaunchTime', 'Unknown').isoformat() if instance.get('LaunchTime') != 'Unknown' else 'Unknown',
                            'availability_zone': instance['Placement']['AvailabilityZone'],
                            'vpc_id': instance.get('VpcId', 'N/A'),
                            'subnet_id': instance.get('SubnetId', 'N/A'),
                            'public_ip': instance.get('PublicIpAddress', 'N/A'),
                            'private_ip': instance.get('PrivateIpAddress', 'N/A'),
                            'security_groups': [
                                {
                                    'group_id': sg['GroupId'],
                                    'group_name': sg['GroupName']
                                }
                                for sg in instance.get('SecurityGroups', [])
                            ],
                            'tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                        }
                        
                        instances.append(instance_info)
            
            if len(instances) == 0:
                print(f"  [WARNING] No EC2 instances found in {self.region}")
            else:
                print(f"  Found {len(instances)} EC2 instances")
            return instances
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'UnauthorizedOperation':
                print(f"  [WARNING] Access denied for EC2 operations - skipping instance enumeration")
                return []
            elif 'Timeout' in str(e) or 'timeout' in str(e).lower():
                print(f"  [WARNING] Network timeout for EC2 operations - retrying once...")
                try:
                    return self.inspect_ec2_instances()
                except:
                    print(f"  [WARNING] Retry failed for EC2 operations - skipping")
                    return []
            else:
                print(f"  [ERROR] Failed to access EC2 operations: {e}")
                return []
    
    def inspect_s3_buckets(self) -> List[Dict[str, Any]]:
        print("🔍 Inspecting S3 buckets...")
        
        try:
            s3 = self._get_client('s3')
            buckets = []
            
            response = s3.list_buckets()
            
            for bucket in response['Buckets']:
                bucket_name = bucket['Name']
                bucket_info = {
                    'bucket_name': bucket_name,
                    'creation_date': bucket['CreationDate'].isoformat(),
                    'region': 'Unknown',
                    'object_count': 0,
                    'size_bytes': 0
                }
                
                try:
                    location = s3.get_bucket_location(Bucket=bucket_name)
                    bucket_info['region'] = location['LocationConstraint'] or 'us-east-1'
                    
                    try:
                        paginator = s3.get_paginator('list_objects_v2')
                        page_iterator = paginator.paginate(Bucket=bucket_name)
                        
                        object_count = 0
                        total_size = 0
                        
                        for page in page_iterator:
                            if 'Contents' in page:
                                object_count += len(page['Contents'])
                                total_size += sum(obj['Size'] for obj in page['Contents'])
                        
                        bucket_info['object_count'] = object_count
                        bucket_info['size_bytes'] = total_size
                        
                    except ClientError as e:
                        if e.response['Error']['Code'] == 'AccessDenied':
                            bucket_info['object_count'] = 'Access Denied'
                            bucket_info['size_bytes'] = 'Access Denied'
                
                except ClientError as e:
                    error_code = e.response['Error']['Code']
                    if error_code == 'AccessDenied':
                        print(f"  [ERROR] Failed to access S3 bucket '{bucket_name}': Access Denied")
                        continue
                    elif 'Timeout' in str(e) or 'timeout' in str(e).lower():
                        print(f"  [WARNING] Network timeout accessing bucket '{bucket_name}' - skipping")
                        continue
                
                buckets.append(bucket_info)
            
            print(f"  Found {len(buckets)} S3 buckets")
            return buckets
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'AccessDenied':
                print(f"  [WARNING] Access denied for S3 operations - skipping bucket enumeration")
                return []
            elif 'Timeout' in str(e) or 'timeout' in str(e).lower():
                print(f"  [WARNING] Network timeout for S3 operations - retrying once...")
                try:
                    return self.inspect_s3_buckets()
                except:
                    print(f"  [WARNING] Retry failed for S3 operations - skipping")
                    return []
            else:
                print(f"  [ERROR] Failed to access S3 operations: {e}")
                return []
    
    def inspect_security_groups(self) -> List[Dict[str, Any]]:
        print("🔍 Inspecting security groups...")
        
        try:
            ec2 = self._get_client('ec2')
            security_groups = []
            
            paginator = ec2.get_paginator('describe_security_groups')
            for page in paginator.paginate():
                for sg in page['SecurityGroups']:
                    sg_info = {
                        'group_id': sg['GroupId'],
                        'group_name': sg['GroupName'],
                        'description': sg['Description'],
                        'vpc_id': sg.get('VpcId', 'N/A'),
                        'inbound_rules': [],
                        'outbound_rules': []
                    }
                    
                    for rule in sg.get('IpPermissions', []):
                        rule_info = {
                            'protocol': rule.get('IpProtocol', 'All'),
                            'from_port': rule.get('FromPort', 'All'),
                            'to_port': rule.get('ToPort', 'All'),
                            'sources': []
                        }
                        
                        for ip_range in rule.get('IpRanges', []):
                            rule_info['sources'].append({
                                'type': 'CIDR',
                                'value': ip_range['CidrIp'],
                                'description': ip_range.get('Description', '')
                            })
                        
                        for sg_ref in rule.get('UserIdGroupPairs', []):
                            rule_info['sources'].append({
                                'type': 'SecurityGroup',
                                'value': sg_ref['GroupId'],
                                'description': sg_ref.get('Description', '')
                            })
                        
                        sg_info['inbound_rules'].append(rule_info)
                    
                    for rule in sg.get('IpPermissionsEgress', []):
                        rule_info = {
                            'protocol': rule.get('IpProtocol', 'All'),
                            'from_port': rule.get('FromPort', 'All'),
                            'to_port': rule.get('ToPort', 'All'),
                            'destinations': []
                        }
                        
                        for ip_range in rule.get('IpRanges', []):
                            rule_info['destinations'].append({
                                'type': 'CIDR',
                                'value': ip_range['CidrIp'],
                                'description': ip_range.get('Description', '')
                            })
                        
                        for sg_ref in rule.get('UserIdGroupPairs', []):
                            rule_info['destinations'].append({
                                'type': 'SecurityGroup',
                                'value': sg_ref['GroupId'],
                                'description': sg_ref.get('Description', '')
                            })
                        
                        sg_info['outbound_rules'].append(rule_info)
                    
                    security_groups.append(sg_info)
            
            print(f"  Found {len(security_groups)} security groups")
            return security_groups
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'UnauthorizedOperation':
                print(f"  [WARNING] Access denied for security group operations - skipping enumeration")
                return []
            elif 'Timeout' in str(e) or 'timeout' in str(e).lower():
                print(f"  [WARNING] Network timeout for security group operations - retrying once...")
                try:
                    return self.inspect_security_groups()
                except:
                    print(f"  [WARNING] Retry failed for security group operations - skipping")
                    return []
            else:
                print(f"  [ERROR] Failed to access security group operations: {e}")
                return []
    
    def inspect_all(self) -> Dict[str, Any]:
        print("Starting AWS resource inspection...")
        print("=" * 60)
        
        account_info = self._get_account_info()
        
        iam_users = self.inspect_iam_users()
        ec2_instances = self.inspect_ec2_instances()
        s3_buckets = self.inspect_s3_buckets()
        security_groups = self.inspect_security_groups()
        
        summary = {
            'total_users': len(iam_users),
            'running_instances': len([i for i in ec2_instances if i.get('state') == 'running']),
            'total_buckets': len(s3_buckets),
            'security_groups': len(security_groups)
        }
        
        inspection_results = {
            'account_info': account_info,
            'resources': {
                'iam_users': iam_users,
                'ec2_instances': ec2_instances,
                's3_buckets': s3_buckets,
                'security_groups': security_groups
            },
            'summary': summary
        }
        
        print("=" * 60)
        print("Inspection completed!")
        
        return inspection_results
    
    def _get_account_info(self) -> Dict[str, Any]:
        try:
            sts = self._get_client('sts')
            identity = sts.get_caller_identity()
            
            return {
                'account_id': identity.get('Account', 'Unknown'),
                'user_arn': identity.get('Arn', 'Unknown'),
                'region': self.region,
                'scan_timestamp': datetime.utcnow().isoformat() + 'Z'
            }
        except Exception as e:
            return {
                'account_id': 'Unknown',
                'user_arn': 'Unknown', 
                'region': self.region,
                'scan_timestamp': datetime.utcnow().isoformat() + 'Z'
            }


def format_as_table(data: Dict[str, Any]) -> str:
    output = []
    output.append("AWS RESOURCE INSPECTION REPORT")
    output.append("=" * 50)
    output.append(f"Account ID: {data['account_info']['account_id']}")
    output.append(f"User ARN: {data['account_info']['user_arn']}")
    output.append(f"Region: {data['account_info']['region']}")
    output.append(f"Scan Timestamp: {data['account_info']['scan_timestamp']}")
    output.append("")
    
    output.append("SUMMARY:")
    output.append("-" * 20)
    output.append(f"  Total IAM Users: {data['summary']['total_users']}")
    output.append(f"  Running EC2 Instances: {data['summary']['running_instances']}")
    output.append(f"  Total S3 Buckets: {data['summary']['total_buckets']}")
    output.append(f"  Security Groups: {data['summary']['security_groups']}")
    output.append("")
    
    output.append("IAM USERS:")
    output.append("-" * 20)
    if data['resources']['iam_users']:
        for user in data['resources']['iam_users']:
            output.append(f"  • {user['username']} ({user['user_id']})")
            output.append(f"    Created: {user['created_date']}")
            output.append(f"    Last Password Use: {user['password_last_used']}")
            if user['attached_policies']:
                output.append(f"    Attached Policies: {len(user['attached_policies'])}")
            if user['inline_policies']:
                output.append(f"    Inline Policies: {len(user['inline_policies'])}")
            output.append("")
    else:
        output.append("  No IAM users found or insufficient permissions")
        output.append("")
    
    output.append("EC2 INSTANCES:")
    output.append("-" * 20)
    if data['resources']['ec2_instances']:
        for instance in data['resources']['ec2_instances']:
            output.append(f"  • {instance['instance_id']} ({instance['name']})")
            output.append(f"    Type: {instance['instance_type']}")
            output.append(f"    State: {instance['state']}")
            output.append(f"    AZ: {instance['availability_zone']}")
            if instance['public_ip'] != 'N/A':
                output.append(f"    Public IP: {instance['public_ip']}")
            output.append(f"    Private IP: {instance['private_ip']}")
            output.append("")
    else:
        output.append("  No EC2 instances found")
        output.append("")
    
    output.append("S3 BUCKETS:")
    output.append("-" * 20)
    if data['resources']['s3_buckets']:
        for bucket in data['resources']['s3_buckets']:
            output.append(f"  • {bucket['bucket_name']}")
            output.append(f"    Region: {bucket['region']}")
            output.append(f"    Created: {bucket['creation_date']}")
            output.append(f"    Object Count: {bucket['object_count']}")
            if isinstance(bucket['size_bytes'], int):
                size_mb = bucket['size_bytes'] / (1024 * 1024)
                output.append(f"    Size: {size_mb:.2f} MB")
            else:
                output.append(f"    Size: {bucket['size_bytes']}")
            output.append("")
    else:
        output.append("  No S3 buckets found")
        output.append("")
    
    output.append("SECURITY GROUPS:")
    output.append("-" * 20)
    if data['resources']['security_groups']:
        for sg in data['resources']['security_groups']:
            output.append(f"  • {sg['group_id']} ({sg['group_name']})")
            output.append(f"    Description: {sg['description']}")
            output.append(f"    VPC: {sg['vpc_id']}")
            output.append(f"    Inbound Rules: {len(sg['inbound_rules'])}")
            output.append(f"    Outbound Rules: {len(sg['outbound_rules'])}")
            output.append("")
    else:
        output.append("  No security groups found")
        output.append("")
    
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description='AWS Resource Inspector - List and inspect AWS resources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aws_inspector.py
  python aws_inspector.py --region us-west-2
  python aws_inspector.py --output report.json --format json
  python aws_inspector.py --region eu-west-1 --output report.txt --format table
        """
    )
    
    parser.add_argument('--region', 
                       help='AWS region to inspect (default: from credentials/config)')
    parser.add_argument('--output', 
                       help='Output file path (default: print to stdout)')
    parser.add_argument('--format', 
                       choices=['json', 'table'], 
                       default='json',
                       help='Output format (default: json)')
    
    args = parser.parse_args()
    
    inspector = AWSInspector(region=args.region)
    
    if not inspector.authenticate():
        sys.exit(1)
    
    print("")
    
    results = inspector.inspect_all()
    
    if args.format == 'json':
        output = json.dumps(results, indent=2, default=str)
    else:
        output = format_as_table(results)
    
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"\nReport saved to: {args.output}")
        except Exception as e:
            print(f"Error writing to file: {e}")
            sys.exit(1)
    else:
        print("\n" + "=" * 60)
        print("INSPECTION RESULTS:")
        print("=" * 60)
        print(output)


if __name__ == "__main__":
    main()
