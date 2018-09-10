#include<iostream>
#include<vector>
using namespace std;

int lis(int a[], int distance[], int N)
{
	int best = 0;
	for (int i = 0; i < N; i++)
		for (int j = i-1; j >= 0; j--)
		{
			if (a[i] >= a[j]&&distance[j]+1>distance[i])
			{
				distance[i] = distance[j] + 1;
			}
		}

	best = distance[0];
	for (int i = 0; i < N; i++)
		if (distance[i]>best)
			best = distance[i];

	return best;
}

int BinarySearch(vector<int> &a, int k)
{
	int i = 0;
	int low = 0, high = a.size() - 1;
	while (low < high)
	{
		int mid = (low + high) / 2;
		if (a[mid] == k)
		{
			i = mid;
			break;
		}
		else if (a[mid] > k)
			high = mid - 1;
		else
			low = mid + 1;
	}
	return low;
}

int LIS(vector<int> nums)
{
	int best = 0;
	int N = nums.size();
	vector<int> distance;
	
	distance.push_back(nums[0]);
	for (int i = 1; i < nums.size(); i++)
	{
		if (nums[i] > distance[distance.size() - 1])
			distance.push_back(nums[i]);
		else
		{
			int pos = BinarySearch(distance, nums[i]);
			distance[pos] = nums[i];
		}
	}
	return distance.size();
}

int main()
{
	/*int *a, *distance;
	int N;
	cin >> N;
	a = new int[N];
	distance = new int[N];
	for (int i = 0; i < N; i++)
	{
		distance[i] = 1;
		cin >> a[i];
	}
	int result = lis(a, distance, N);
	cout << result << endl;*/

	int n;
	cin >> n;
	vector<int> ve(n);
	for (int i = 0; i < n; i++)
	{
		cin >> ve[i];
	}
	int low = LIS(ve);
	cout << low << endl;	
	system("pause");
	return 0;
}