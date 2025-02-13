import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Protocol, Optional
from urllib.parse import urlparse

from opensearchpy import OpenSearch

PERFORMANCE_RECORD_LATENCY_MS = "latency"
PERFORMANCE_RECORD_THROUGHPUT_SAMPLE_PER_SEC = "throughput"


@dataclass
class PerformanceRecord:
    metric: str
    kind: str
    value: Any

    when: datetime = field(default_factory=lambda: datetime.now())
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def latency(metric: str, value_ms: float, meta: Optional[Dict[str, Any]] = None, when: Optional[datetime] = None):
        r"""
        Create a PerformanceRecord tracking latency information
        Args:
            `metric` (`str`):
                Metric identifier
            `value_ms` (`float`):
                The recorded latency, in millisecond, for the underlying metric record
            `meta` (`Optional[Dict[str, Any]]`, defaults to `{}`)
                Information relative to the recorded metric to store alongside the metric readout
            `when` (`Optional[datetime]`, defaults to `datetime.now()`)
                Indicates when the underlying metric was recorded
        Returns:
            The performance record for the target metric representing latency
        """
        return PerformanceRecord(
            metric=metric, kind=PERFORMANCE_RECORD_LATENCY_MS, value=value_ms, when=when, meta=meta
        )

    @staticmethod
    def throughput(metric: str, value_sample_per_sec: float, meta: Optional[Dict[str, Any]] = None,
                   when: Optional[datetime] = None):
        r"""
        Create a PerformanceRecord tracking throughput information
        Args:
            `metric` (`str`):
                Metric identifier
            `value_sample_per_sec` (`float`):
                The recorded throughput, in samples per second, for the underlying metric record
            `meta` (`Optional[Dict[str, Any]]`, defaults to `{}`)
                Information relative to the recorded metric to store alongside the metric readout
            `when` (`Optional[datetime]`, defaults to `datetime.now()`)
                Indicates when the underlying metric was recorded
        Returns:
            The performance record for the target metric representing throughput
        """
        return PerformanceRecord(
            metric=metric,
            kind=PERFORMANCE_RECORD_THROUGHPUT_SAMPLE_PER_SEC,
            value=value_sample_per_sec,
            when=when,
            meta=meta
        )

    def as_document(self) -> Dict[str, Any]:
        r"""
        Convert the actual `PerformanceRecord` to a dictionary based representation compatible with document storage
        Returns:
            Dictionary of strings keys with the information stored in this record
        """
        parcel = {"date": self.when.timestamp(), "metric": self.metric, "kind": self.kind, "value": self.value}
        return parcel | self.meta


class PerformanceTrackerStore(Protocol):
    r"""
    Base interface defining a performance tracker tool
    """

    @staticmethod
    def from_uri(uri: str) -> "PerformanceTrackerStore":
        r"""
        Create the `PerformanceTrackerStore` from the provided URI information
        Args:
         `uri` (`str`):
            URI specifying over which protocol and where will be stored the record(s)
        Returns:
            Instance of a `PerformanceTrackerStore` which information are inferred from the specified URI
        """
        pass

    def push(self, collection: str, record: "PerformanceRecord"):
        r"""
        Attempt to append the provided record to the underlying tracker putting under the specified collection
        Args:
            `collection` (`str`):
                Name of the bucket the specified record should be pushed
            `record` (`PerformanceRecord`):
                The materialized record to push
        """
        pass


class OpenSearchPerformanceTrackerStore(PerformanceTrackerStore):
    r"""
    Amazon Web Services (AWS) OpenSearch based PerformanceTrackerStore
    Supported URIs are as follows:
    - os://<username:password@><hostname>:<port>
    - os+aws://<aws_access_key_id:aws_secret_access_key@><hostname>:<port>
    - os+aws://<hostname>:<port> - will use the stored aws credentials on the system
    """

    # Extract region and service from AWS url (ex: us-east-1.es.amazonaws.com)
    AWS_URL_RE = re.compile(r"([a-z]+-[a-z]+-[0-9])\.(.*)?\.amazonaws.com")

    def __init__(self, url: str, auth):
        uri = urlparse(url)
        self._client = OpenSearch(
            [{"host": uri.hostname, "port": uri.port or 443}],
            http_auth=auth,
            http_compress=True,
            use_ssl=True
        )

        # Sanity check
        self._client.info()

    @staticmethod
    def from_uri(uri: str) -> "PerformanceTrackerStore":
        if not (_uri := urlparse(uri)).scheme.startswith("es"):
            raise ValueError(f"Invalid URI {uri}: should start with os:// or os+aws://")

        if _uri.scheme == "es+aws":
            from boto3 import Session as AwsSession
            from botocore.credentials import Credentials as AwsCredentials
            from opensearchpy import Urllib3AWSV4SignerAuth

            # Create AWS session from the (eventual) creds
            if not _uri.username and not _uri.password:
                session = AwsSession()
                creds = session.get_credentials()
            else:
                creds = AwsCredentials(_uri.username, _uri.password)

            # Parse the url to extract region and service
            if len(match := re.findall(OpenSearchPerformanceTrackerStore.AWS_URL_RE, _uri.netloc)) != 1:
                raise ValueError(f"Failed to parse AWS es service URL {uri}")

            region, service = match[0]
            auth = Urllib3AWSV4SignerAuth(creds, region, service)
        else:
            auth = (_uri.username, _uri.password)

        return OpenSearchPerformanceTrackerStore(uri, auth)

    def _ensure_collection_exists(self, collection: str):
        if not self._client.indices.exists(collection):
            self._client.indices.create(collection)

    def push(self, collection: str, record: "PerformanceRecord"):
        self._ensure_collection_exists(collection)
        self._client.index(collection, record.as_document())


class AutoPerformanceTracker:

    @staticmethod
    def from_uri(uri: str) -> "PerformanceTrackerStore":
        if uri.startswith("es://") or uri.startswith("es+aws://"):
            return OpenSearchPerformanceTrackerStore.from_uri(uri)

        raise ValueError(
            f"Unable to determine the service associated with URI: {uri}. "
            "Valid schemas are es:// or es+aws://"
        )


def main():
    parser = argparse.ArgumentParser(
        prog='text-generation-inference-benchmark-optimum',
        description='Pushes benchmark results to an OpenSearch instance'
    )
    parser.add_argument(
        '--uri',
        type=str,
        required=False,
        help='URI to the OpenSearch instance where to push the benchmark results',
        default='"es+aws://search-optimum-benchmarks-kb3meoztyufprqul537nq7deny.us-east-1.es.amazonaws.com"'
    )
    parser.add_argument(
        '--collection',
        type=str,
        required=False,
        help='Collection name where to push the benchmark results',
        default='ci_tgi_performances_tracker'
    )
    parser.add_argument(
        '--meta',
        action='append',
        required=False,
        help='Meta information to store alongside the benchmark results, use multiple times for multiple values',
        nargs='?'
    )
    parser.add_argument(
        'results',
        type=str,
        help='File containing the benchmark results to push',
    )
    args = parser.parse_args()
    meta = flatten(args.meta)
    bench_id = hashlib.md5(open(args.results, 'rb').read()).hexdigest()
    meta['bench_id'] = bench_id

    with open(args.results, 'r') as f:
        data = json.load(f)

    tracker=AutoPerformanceTracker.from_uri("es+aws://search-optimum-benchmarks-kb3meoztyufprqul537nq7deny.us-east-1.es.amazonaws.com")
    filtered_results = [result for result in data['results'] if
                        result['id'] != 'warmup' and result['id'] != 'throughput']
    latency_metrics_to_push = ['inter_token_latency_ms_p90', 'time_to_first_token_ms_p90', 'e2e_latency_ms_p90']
    throughput_metrics_to_push = ['token_throughput_secs']
    start_time = data['start_time']
    for result in filtered_results:
        for metric in latency_metrics_to_push:
            record = PerformanceRecord.latency(metric, result[metric], {**meta, 'qps': result['config']['rate']},
                                               when=start_time)
            print(record)
            tracker.push("ci_tgi_performances_tracker", record)
        for metric in throughput_metrics_to_push:
            record = PerformanceRecord.throughput(metric, result[metric], {**meta, 'qps': result['config']['rate']},
                                                  when=start_time)
            print(record)
            tracker.push("ci_tgi_performances_tracker", record)

    # record=PerformanceRecord.latency("TIME_TO_FIRST_TOKEN", 100,{})


def flatten(l: list[str]) -> dict[str, str]:
    d = {}
    for e in l:
        e = e.split('=')
        d[e[0]] = e[1]
    return d


if __name__ == '__main__':
    main()
