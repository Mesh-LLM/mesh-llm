//! Users and organizations: identity, profile lookup, and social listings.
//!
//! This module exposes the [`User`], [`OrgMembership`], and [`Organization`]
//! types and the corresponding [`HFClient`] methods:
//!
//! - [`HFClient::whoami`] — identify the caller and verify that the current token is valid.
//! - [`HFClient::user_overview`] / [`HFClient::organization_overview`] — fetch a public profile by username or
//!   organization name.
//! - [`HFClient::list_user_followers`] / [`HFClient::list_user_following`] / [`HFClient::list_organization_members`] —
//!   paginated listings that yield [`User`] entries one page at a time.

use bon::bon;
use futures::Stream;
use serde::Deserialize;
use url::Url;

use crate::client::HFClient;
use crate::error::HFResult;
use crate::retry;

/// A Hugging Face Hub user account.
///
/// Returned by [`HFClient::whoami`] and the various user-lookup endpoints.
/// Only [`username`](Self::username) is guaranteed to be set; the remaining
/// fields are populated for the authenticated caller's own `whoami` response
/// or when the field is publicly visible on the target user's profile.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct User {
    /// Hub handle (slug) of the user — the name used in URLs such as
    /// `https://huggingface.co/<username>`.
    #[serde(alias = "login", alias = "user", alias = "name")]
    pub username: String,
    /// Display name as shown on the user's profile, when set.
    pub fullname: Option<String>,
    /// URL to the user's avatar image.
    pub avatar_url: Option<String>,
    /// Account type, typically `"user"` or `"org"`.
    #[serde(rename = "type")]
    pub user_type: Option<String>,
    /// Free-text bio shown on the user's profile.
    pub details: Option<String>,
    /// Whether the authenticated caller follows this user.
    pub is_following: Option<bool>,
    /// Whether the user is on a Pro plan.
    pub is_pro: Option<bool>,
    /// Number of models created by the user.
    pub num_models: Option<u64>,
    /// Number of datasets created by the user.
    pub num_datasets: Option<u64>,
    /// Number of Spaces created by the user.
    pub num_spaces: Option<u64>,
    /// Number of discussions initiated by the user.
    pub num_discussions: Option<u64>,
    /// Number of papers authored by the user.
    pub num_papers: Option<u64>,
    /// Upvotes the user has received.
    pub num_upvotes: Option<u64>,
    /// Likes the user has given.
    pub num_likes: Option<u64>,
    /// Number of users this user is following.
    pub num_following: Option<u64>,
    /// Number of users following this user.
    pub num_followers: Option<u64>,
    /// Email address — only returned by `whoami` for the authenticated user.
    pub email: Option<String>,
    /// Whether the email has been verified — only returned by `whoami`.
    pub email_verified: Option<bool>,
    /// Billing plan identifier — only returned by `whoami`.
    pub plan: Option<String>,
    /// Whether the account has a valid payment method — only returned by `whoami`.
    pub can_pay: Option<bool>,
    /// Organizations the authenticated user belongs to. Only populated by
    /// `whoami` for the caller themselves.
    pub orgs: Option<Vec<OrgMembership>>,
}

/// Summary entry for an organization the authenticated user belongs to.
///
/// Returned inside [`User::orgs`]. This is a lighter-weight shape than
/// [`Organization`] — use [`HFClient::organization_overview`] to fetch the
/// full record by name.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrgMembership {
    /// Hub handle (slug) of the organization.
    pub name: Option<String>,
    /// Display name as shown on the organization's profile.
    pub fullname: Option<String>,
    /// URL to the organization's avatar image.
    pub avatar_url: Option<String>,
}

/// A Hugging Face Hub organization.
///
/// Returned by [`HFClient::organization_overview`].
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Organization {
    /// Hub handle (slug) of the organization — the name used in URLs such as
    /// `https://huggingface.co/<name>`.
    pub name: String,
    /// Display name as shown on the organization's profile, when set.
    pub fullname: Option<String>,
    /// URL to the organization's avatar image.
    pub avatar_url: Option<String>,
    /// Account type, typically `"org"`.
    #[serde(rename = "type")]
    pub org_type: Option<String>,
    /// Free-text description shown on the organization's profile.
    pub details: Option<String>,
    /// Whether the organization is verified.
    pub is_verified: Option<bool>,
    /// Whether the authenticated caller follows this organization.
    pub is_following: Option<bool>,
    /// Number of members in the organization.
    pub num_users: Option<u64>,
    /// Number of models owned by the organization.
    pub num_models: Option<u64>,
    /// Number of Spaces owned by the organization.
    pub num_spaces: Option<u64>,
    /// Number of datasets owned by the organization.
    pub num_datasets: Option<u64>,
    /// Number of followers of the organization.
    pub num_followers: Option<u64>,
    /// Number of papers authored by the organization.
    pub num_papers: Option<u64>,
    /// Plan identifier (e.g., `"enterprise"`, `"team"`).
    pub plan: Option<String>,
}

#[bon]
impl HFClient {
    /// Fetch the profile of the user that owns the current token.
    ///
    /// Returns the authenticated [`User`], including private fields like
    /// [`email`](User::email) and the caller's [`orgs`](User::orgs) list. Fails with
    /// [`HFError::AuthRequired`](crate::HFError::AuthRequired) if no valid token is configured.
    ///
    /// Endpoint: `GET /api/whoami-v2`.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn whoami(&self) -> HFResult<User> {
        let url = format!("{}/api/whoami-v2", self.endpoint());
        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client().get(&url).headers(headers.clone()).send()
        })
        .await?;
        let response = self
            .check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// Fetch the public profile of a user by Hub handle.
    ///
    /// Endpoint: `GET /api/users/{username}/overview`.
    ///
    /// # Parameters
    ///
    /// - `username` (required): Hub handle (slug) of the user.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn user_overview(
        &self,
        /// Hub handle (slug) of the user.
        username: &str,
    ) -> HFResult<User> {
        let url = format!("{}/api/users/{}/overview", self.endpoint(), username);
        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client().get(&url).headers(headers.clone()).send()
        })
        .await?;
        let response = self
            .check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// Fetch the public profile of an organization by Hub handle.
    ///
    /// Endpoint: `GET /api/organizations/{organization}/overview`.
    ///
    /// # Parameters
    ///
    /// - `organization` (required): Hub handle (slug) of the organization.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub async fn organization_overview(
        &self,
        /// Hub handle (slug) of the organization.
        organization: &str,
    ) -> HFResult<Organization> {
        let url = format!(
            "{}/api/organizations/{}/overview",
            self.endpoint(),
            organization
        );
        let headers = self.auth_headers();
        let response = retry::retry(self.retry_config(), || {
            self.http_client().get(&url).headers(headers.clone()).send()
        })
        .await?;
        let response = self
            .check_response(response, None, crate::error::NotFoundContext::Generic)
            .await?;
        Ok(response.json().await?)
    }

    /// Stream the followers of a user.
    ///
    /// Endpoint: `GET /api/users/{username}/followers`.
    ///
    /// # Parameters
    ///
    /// - `username` (required): Hub handle of the user.
    /// - `limit`: cap on the total number of items yielded.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_user_followers(
        &self,
        /// Hub handle of the user.
        #[builder(into)]
        username: String,
        /// Cap on the total number of items yielded.
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<User>> + '_> {
        let url = Url::parse(&format!(
            "{}/api/users/{}/followers",
            self.endpoint(),
            username
        ))?;
        Ok(self.paginate(url, vec![], limit))
    }

    /// Stream the users that a user is following.
    ///
    /// Endpoint: `GET /api/users/{username}/following`.
    ///
    /// # Parameters
    ///
    /// - `username` (required): Hub handle of the user.
    /// - `limit`: cap on the total number of items yielded.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_user_following(
        &self,
        /// Hub handle of the user.
        #[builder(into)]
        username: String,
        /// Cap on the total number of items yielded.
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<User>> + '_> {
        let url = Url::parse(&format!(
            "{}/api/users/{}/following",
            self.endpoint(),
            username
        ))?;
        Ok(self.paginate(url, vec![], limit))
    }

    /// Stream the members of an organization.
    ///
    /// Endpoint: `GET /api/organizations/{organization}/members`.
    ///
    /// # Parameters
    ///
    /// - `organization` (required): Hub handle of the organization.
    /// - `limit`: cap on the total number of items yielded.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_organization_members(
        &self,
        /// Hub handle of the organization.
        #[builder(into)]
        organization: String,
        /// Cap on the total number of items yielded.
        limit: Option<usize>,
    ) -> HFResult<impl Stream<Item = HFResult<User>> + '_> {
        let url = Url::parse(&format!(
            "{}/api/organizations/{}/members",
            self.endpoint(),
            organization
        ))?;
        Ok(self.paginate(url, vec![], limit))
    }
}

#[cfg(feature = "blocking")]
#[bon]
impl crate::blocking::HFClientSync {
    /// Blocking counterpart of [`HFClient::whoami`].
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn whoami(&self) -> HFResult<User> {
        self.runtime.block_on(self.inner.whoami().send())
    }

    /// Blocking counterpart of [`HFClient::user_overview`]. See the async method for parameters
    /// and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn user_overview(&self, username: &str) -> HFResult<User> {
        self.runtime
            .block_on(self.inner.user_overview().username(username).send())
    }

    /// Blocking counterpart of [`HFClient::organization_overview`]. See the async method for
    /// parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn organization_overview(&self, organization: &str) -> HFResult<Organization> {
        self.runtime.block_on(
            self.inner
                .organization_overview()
                .organization(organization)
                .send(),
        )
    }

    /// Blocking counterpart of [`HFClient::list_user_followers`]. Collects the stream into a
    /// `Vec<User>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_user_followers(
        &self,
        #[builder(into)] username: String,
        limit: Option<usize>,
    ) -> HFResult<Vec<User>> {
        use futures::StreamExt;
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_user_followers()
                .username(username)
                .maybe_limit(limit)
                .send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFClient::list_user_following`]. Collects the stream into a
    /// `Vec<User>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_user_following(
        &self,
        #[builder(into)] username: String,
        limit: Option<usize>,
    ) -> HFResult<Vec<User>> {
        use futures::StreamExt;
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_user_following()
                .username(username)
                .maybe_limit(limit)
                .send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }

    /// Blocking counterpart of [`HFClient::list_organization_members`]. Collects the stream into a
    /// `Vec<User>`. See the async method for parameters and behavior.
    #[builder(finish_fn = send, derive(Debug, Clone))]
    pub fn list_organization_members(
        &self,
        #[builder(into)] organization: String,
        limit: Option<usize>,
    ) -> HFResult<Vec<User>> {
        use futures::StreamExt;
        self.runtime.block_on(async move {
            let stream = self
                .inner
                .list_organization_members()
                .organization(organization)
                .maybe_limit(limit)
                .send()?;
            futures::pin_mut!(stream);
            let mut items = Vec::new();
            while let Some(item) = stream.next().await {
                items.push(item?);
            }
            Ok(items)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{Organization, User};

    #[test]
    fn test_user_full_profile() {
        let json = r#"{
            "user":"alice",
            "fullname":"Alice Anderson",
            "avatarUrl":"https://example/a.png",
            "type":"user",
            "details":"researcher",
            "isFollowing":true,
            "isPro":false,
            "numModels":12,
            "numDatasets":3,
            "numSpaces":1,
            "numDiscussions":4,
            "numPapers":2,
            "numUpvotes":5,
            "numLikes":6,
            "numFollowing":7,
            "numFollowers":8
        }"#;
        let user: User = serde_json::from_str(json).unwrap();
        assert_eq!(user.username, "alice");
        assert_eq!(user.details.as_deref(), Some("researcher"));
        assert_eq!(user.is_following, Some(true));
        assert_eq!(user.num_models, Some(12));
        assert_eq!(user.num_datasets, Some(3));
        assert_eq!(user.num_spaces, Some(1));
        assert_eq!(user.num_discussions, Some(4));
        assert_eq!(user.num_papers, Some(2));
        assert_eq!(user.num_upvotes, Some(5));
        assert_eq!(user.num_likes, Some(6));
        assert_eq!(user.num_following, Some(7));
        assert_eq!(user.num_followers, Some(8));
    }

    #[test]
    fn test_organization_full_profile() {
        let json = r#"{
            "name":"acme",
            "fullname":"Acme Corp",
            "avatarUrl":"https://example/o.png",
            "type":"org",
            "details":"description",
            "isVerified":true,
            "isFollowing":false,
            "numUsers":42,
            "numModels":7,
            "numSpaces":2,
            "numDatasets":3,
            "numFollowers":100,
            "numPapers":5,
            "plan":"enterprise"
        }"#;
        let org: Organization = serde_json::from_str(json).unwrap();
        assert_eq!(org.name, "acme");
        assert_eq!(org.details.as_deref(), Some("description"));
        assert_eq!(org.is_verified, Some(true));
        assert_eq!(org.is_following, Some(false));
        assert_eq!(org.num_users, Some(42));
        assert_eq!(org.num_models, Some(7));
        assert_eq!(org.num_spaces, Some(2));
        assert_eq!(org.num_datasets, Some(3));
        assert_eq!(org.num_followers, Some(100));
        assert_eq!(org.num_papers, Some(5));
        assert_eq!(org.plan.as_deref(), Some("enterprise"));
    }
}
