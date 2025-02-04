resource "aws_s3_bucket" "cz_benchmarks" {
  bucket = "cz-benchmarks"
}

# Configure public access settings for the bucket
# All settings must be false to allow public read access
resource "aws_s3_bucket_public_access_block" "public_read_access" {
  bucket = aws_s3_bucket.cz_benchmarks.id

  # Allow creation of public ACLs for the bucket and objects
  block_public_acls = false

  # Allow public bucket policies to be set on the bucket
  # Must be false for our public read policy to work
  block_public_policy = false

  # Honor any public ACLs set on the bucket and objects
  ignore_public_acls = false

  # Allow the bucket to be public
  # Must be false to allow any public access
  restrict_public_buckets = false
}

# Policy that grants public read access to all objects in the bucket
# This allows anyone to download/view objects without authentication
resource "aws_s3_bucket_policy" "public_read_policy" {
  bucket = aws_s3_bucket.cz_benchmarks.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "PublicReadGetObject"
        Effect    = "Allow"
        # "*" means anyone can access
        Principal = "*"
        # s3:GetObject allows downloading/viewing objects
        Action    = "s3:GetObject"
        # Apply to all objects in the bucket
        Resource  = "${aws_s3_bucket.cz_benchmarks.arn}/*"
      }
    ]
  })

  # Wait for public access block settings before applying this policy
  depends_on = [aws_s3_bucket_public_access_block.public_read_access]
}

# IAM policy defining write permissions for the bucket
resource "aws_iam_policy" "s3_write_policy" {
  name        = "s3-bucket-write-policy"
  description = "Policy for write access to specific S3 bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          # Grant access to bucket-level operations (for ListBucket)
          aws_s3_bucket.cz_benchmarks.arn,
          # Grant access to object-level operations (for Put/Delete)
          "${aws_s3_bucket.cz_benchmarks.arn}/*"
        ]
      }
    ]
  })
}

# Add policy to allow poweruser role write access to the bucket
resource "aws_iam_role_policy_attachment" "poweruser_bucket_access" {
  role       = "poweruser"
  policy_arn = aws_iam_policy.s3_write_policy.arn
}