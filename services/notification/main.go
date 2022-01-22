package main

import (
	"notification/internal"
	"os"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func init() {
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})
}

func main() {
	env := internal.GetEnv()

	pubsub := internal.NewPubSub(env.NatsUrl, env.NatsUser, env.NatsPass)
	defer pubsub.Close()

	telegram := internal.NewTelegramBot(env.TelegramApiToken, env.TelegramChatId)

	pubsub.Subscribe(internal.NotifyTradeEvent, func(p internal.NotifyTradeEventPayload) {
		message := telegram.FormatTradeMessage(p)
		telegram.SendMessage(internal.NotifyTradeEvent, message)
	})

	pubsub.Subscribe(internal.CriticalErrorEvent, func(p internal.CriticalErrorEventPayload) {
		message := telegram.FormatErrorMessage(p)
		telegram.SendMessage(internal.CriticalErrorEvent, message)
	})

	telegram.ListenForCommands()
}