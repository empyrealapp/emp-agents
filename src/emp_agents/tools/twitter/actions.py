import os
import tempfile
from typing import Annotated

import httpx
import tweepy
from tweepy.asynchronous import AsyncClient
from typing_extensions import Doc

from emp_agents.logger import logger


def _get_env():
    api_key, api_key_secret, access_token, access_token_secret, bearer_token = (
        os.environ.get("X_API_KEY"),
        os.environ.get("X_API_KEY_SECRET"),
        os.environ.get("X_ACCESS_TOKEN"),
        os.environ.get("X_ACCESS_TOKEN_SECRET"),
        os.environ.get("X_BEARER_TOKEN"),
    )
    return api_key, api_key_secret, access_token, access_token_secret, bearer_token


def get_twitter_conn_v1() -> tweepy.API:
    """Get twitter conn 1.1"""

    (
        api_key,
        api_key_secret,
        access_token,
        access_token_secret,
        bearer_token,
    ) = _get_env()

    auth = tweepy.OAuthHandler(
        api_key,
        api_key_secret,
        access_token,
        access_token_secret,
    )
    return tweepy.API(auth)


def _make_client() -> AsyncClient:
    (
        bearer_token,
        api_key,
        api_key_secret,
        access_token,
        access_token_secret,
    ) = _get_env()
    return AsyncClient(
        bearer_token=bearer_token,
        consumer_key=api_key,
        consumer_secret=api_key_secret,
        access_token=access_token,
        access_token_secret=access_token_secret,
    )


async def make_poll(
    text: Annotated[str, Doc("The content of the tweet to be made.")],
    duration_minutes: Annotated[int, Doc("The duration of the poll in minutes.")],
    options: Annotated[list[str], Doc("The options for the poll.")],
):
    """
    Make a poll.  Returns "poll submitted" if successful, or if there is an error, returns the error message.
    """
    api = _make_client()
    await api.create_tweet(
        text=text, duration_minutes=duration_minutes, poll_options=options
    )


async def make_tweet_with_image(
    content: Annotated[str, Doc("The content of the tweet to be made.")],
    image_url: Annotated[str, Doc("The URL of the image to be uploaded.")],
):
    """
    Make a tweet with an image.  Returns "tweet submitted" if successful, or if there is an error, returns the error message.
    """
    api = _make_client()

    v1_client = get_twitter_conn_v1()
    async with httpx.AsyncClient() as client:
        response = await client.get(image_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(response.content)
        image_path = temp_file.name

    media = v1_client.media_upload(image_path)
    media_id = media.media_id

    response = await api.create_tweet(
        text=content,
        media_ids=[media_id],
    )
    return f"tweet submitted: {response}"


async def make_tweet(
    content: Annotated[str, Doc("The content of the tweet to be made.")]
):
    """
    Make a tweet.  Returns "tweet submitted" if successful, or if there is an error, returns the error message.
    """
    api = _make_client()
    response = await api.create_tweet(
        text=content,
    )
    return f"tweet submitted: {response}"


async def make_thread(
    content_list: Annotated[list[str], Doc("List of tweets for a thread")],
):
    """
    Make a thread of tweets on twitter
    """

    if len(content_list) == 0:
        return "no content to make a thread"

    if len(content_list) > 6:
        return "too many tweets for a thread.  Make sure its 6 or less."

    api = _make_client()
    tweet = await api.create_tweet(
        text=content_list[0],
    )

    tweet_id = tweet.data["id"]
    for content in content_list[1:]:
        tweet = await api.create_tweet(
            text=content,
            in_reply_to_tweet_id=tweet_id,
        )
        tweet_id = tweet.data["id"]
    return "X thread submitted"


async def reply_to_tweet(
    tweet_id: Annotated[int, Doc("The ID of the tweet to reply to.")],
    content: Annotated[str, Doc("The content of the tweet to be made.")],
):
    """
    Reply to a tweet.  Returns "tweet submitted" if successful, or if there is an error, returns the error message.
    """
    api = _make_client()
    try:
        tweet = await api.create_tweet(
            text=content,
            in_reply_to_tweet_id=tweet_id,
        )
    except Exception as e:
        logger.error(f"Error submitting tweet: {e}")
        return "error submitting tweet"

    tweet_id = tweet.data["id"]
    return f"id: {tweet_id} | tweet submitted"
